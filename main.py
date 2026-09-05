from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

import requests

from client_app.persistent_worker import (
    run_client_round_task,
)
from client_app.api_client import (
    send_update,
    set_server_url,
)
from config import (
    DIRICHLET_ALPHA,
    CLIENT_WORKERS,
    MODEL_CHANGE_THRESHOLD,
    NUM_ROUNDS,
    PATIENCE,
    SERVER_URL,
    VAL_LOSS_CHANGE_THRESHOLD,
    get_malicious_ids,
)
from evaluation.csv_output import save_round_to_csv


SERVER = SERVER_URL



# Run selected clients using persistent processes


def train_main_and_backup_clients(
    executor,
    main_ids,
    backup_ids,
):

    main_ids = sorted(
        set(main_ids)
    )

    backup_ids = sorted(
        set(backup_ids)
        - set(main_ids)
    )


    main_futures = {}
    backup_futures = {}


    
    # Submit Main clients
    

    for client_id in main_ids:

        future = executor.submit(
            run_client_round_task,
            client_id,
            SERVER,
            True,
        )

        main_futures[
            client_id
        ] = future


    
    # Submit Backup clients
    

    for client_id in backup_ids:

        future = executor.submit(
            run_client_round_task,
            client_id,
            SERVER,
            False,
        )

        backup_futures[
            client_id
        ] = future


    
    # Wait ONLY for Main clients
    

    for client_id, future in (
        main_futures.items()
    ):

        result = future.result()


        if (
            result["client_id"]
            != client_id
        ):
            raise RuntimeError(
                "Client result mismatch."
            )


        if not result["sent"]:

            raise RuntimeError(
                f"Main client {client_id} "
                "did not send its update."
            )


    return backup_futures

def send_required_backup_updates(
    backup_requirements,
    backup_futures,
    round_id,
):

    required_ids = set()


    for client_ids in (
        backup_requirements.values()
    ):

        required_ids.update(
            client_ids
        )


    for client_id in sorted(
        required_ids
    ):

        if (
            client_id
            not in backup_futures
        ):

            raise RuntimeError(
                f"Requested backup "
                f"{client_id} was not "
                "scheduled for training."
            )


        future = backup_futures[
            client_id
        ]


        result = future.result()


        if (
            result["client_id"]
            != client_id
        ):

            raise RuntimeError(
                "Backup client "
                "result mismatch."
            )


        if result["sent"]:

            raise RuntimeError(
                f"Backup client "
                f"{client_id} sent "
                "its update too early."
            )


        if (
            result["round_id"]
            != round_id
        ):

            raise RuntimeError(
                f"Stale backup update: "
                f"client={client_id}, "
                f"cached_round="
                f"{result['round_id']}, "
                f"current_round="
                f"{round_id}"
            )


        response = send_update(
            client_id,
            result["update"],
            round_id=round_id,
        )


        print(
            "Backup update sent:",
            client_id,
            response,
        )



        del backup_futures[
            client_id
        ]


def finish_remaining_backups(
    backup_futures,
    round_id,
):

    for client_id, future in list(
        backup_futures.items()
    ):

        result = future.result()


        if not isinstance(
            result,
            dict,
        ):
            raise RuntimeError(
                f"Invalid backup result "
                f"for client {client_id}"
            )


        if (
            result["client_id"]
            != client_id
        ):
            raise RuntimeError(
                "Backup client "
                "result mismatch."
            )


        if result["sent"]:
            raise RuntimeError(
                f"Backup client "
                f"{client_id} sent "
                "its update unexpectedly."
            )


        if (
            result["round_id"]
            != round_id
        ):
            raise RuntimeError(
                f"Backup round mismatch: "
                f"client={client_id}, "
                f"result_round="
                f"{result['round_id']}, "
                f"expected_round="
                f"{round_id}"
            )


    backup_futures.clear()

# Main
def main():

    set_server_url(
        SERVER
    )
    
    previous_val_loss = None
    stable_checks = 0
    converged = False

    malicious_ids = get_malicious_ids()

    print(
        "Malicious clients:",
        sorted(malicious_ids),
    )


    
    # Clustering ONCE before Round 1
    

    print("\n========================")
    print("INITIALIZING CLUSTERING")
    print("========================")

    response = requests.post(
        f"{SERVER}/initialize_clustering"
    )

    response.raise_for_status()

    clustering_result = response.json()


    if (
        clustering_result.get("status")
        == "waiting"
    ):
        raise RuntimeError(
            "Not enough registered clients "
            "for clustering: "
            f"{clustering_result}"
        )


    if (
        clustering_result.get("status")
        not in (
            "clustering initialized",
            "clustering already initialized",
        )
    ):
        raise RuntimeError(
            "Clustering initialization failed: "
            f"{clustering_result}"
        )


    print(
        "Clustering completed:",
        clustering_result["clusters"],
    )


    
    # Persistent client workers
    

    with ProcessPoolExecutor(
        max_workers=CLIENT_WORKERS
    ) as client_executor:


        
        # Federated rounds
        

        for round_id in range(
            1,
            NUM_ROUNDS + 1,
        ):

            print(
                "\n========================"
            )
            print(
                f"ROUND {round_id}"
            )
            print(
                "========================"
            )


           
            # Start round
           

            response = requests.post(
                f"{SERVER}/start_round",
                json={
                    "round_id": round_id
                },
            )

            response.raise_for_status()

            start_result = response.json()

            print(
                "Round started:",
                start_result,
            )


           
            # Prepare round
            # Only selection happens here.
            # Clustering is NOT repeated.
           

            response = requests.post(
                f"{SERVER}/prepare_round"
            )

            response.raise_for_status()

            prepare_result = response.json()


            if (
                prepare_result.get("status")
                != "round prepared"
            ):
                raise RuntimeError(
                    "Prepare round failed: "
                    f"{prepare_result}"
                )


           
            # Selected clients
           

            response = requests.get(
                f"{SERVER}/round_selection"
            )

            response.raise_for_status()

            selection = response.json()


            main_clients = selection[
                "main_clients"
            ]

            backup_clients = selection[
                "backup_clients"
            ]

            representation_fairness = selection[
                "representation_fairness"
            ]

            hellinger_distance = selection[
                "hellinger_distance"
            ]


            main_ids = []
            backup_ids = []


            for cluster_clients in (
                main_clients.values()
            ):
                main_ids.extend(
                    cluster_clients
                )


            for cluster_clients in (
                backup_clients.values()
            ):
                backup_ids.extend(
                    cluster_clients
                )


            

            


            print(
                "Main clients:",
                main_ids,
            )

            print(
                "Backup clients (if needed):",
                backup_ids,
            )

            print(
                "Selected malicious main clients:",
                sorted(
                    set(main_ids)
                    &
                    malicious_ids
                ),
            )

            print(
                "Selected malicious backup clients:",
                sorted(
                    set(backup_ids)
                    &
                    malicious_ids
                ),
            )

            print(
                f"Representation Fairness: "
                f"{representation_fairness * 100:.2f}%"
            )

            print(
                f"Hellinger Distance: "
                f"{hellinger_distance:.4f}"
            )

            
            # Phase 1
            # Main and backup clients train
            # at the same time.
            #
            # Main updates are sent immediately.
            # Backup updates are cached locally.

            backup_futures = (
                train_main_and_backup_clients(
                    client_executor,
                    main_ids,
                    backup_ids,
                )
            )


            print(
                "\nAll selected main and "
                "backup clients finished training."
            )

            print(
                "Cached backup updates:",
                sorted(
                    backup_futures.keys()
                ),
            )


            
            # Trust evaluation + aggregation
            

            response = requests.post(
                f"{SERVER}/aggregate"
            )

            response.raise_for_status()

            aggregation_result = (
                response.json()
            )

            backup_request_cycles = 0


            while (
                aggregation_result.get(
                    "status"
                )
                == "backup_needed"
            ):

                backup_request_cycles += 1

                if backup_request_cycles > 3:
                    raise RuntimeError(
                        "Too many backup "
                        "request cycles."
                    )


                stage = (
                    aggregation_result.get(
                        "stage"
                    )
                )

                backup_requirements = (
                    aggregation_result[
                        "backup_requirements"
                    ]
                )


                print(
                    "\nBackup updates requested"
                    f" | stage={stage}"
                )

                print(
                    backup_requirements
                )


                send_required_backup_updates(
                    backup_requirements,
                    backup_futures,
                    round_id,
                )


                response = requests.post(
                    f"{SERVER}/aggregate"
                )

                response.raise_for_status()

                aggregation_result = (
                    response.json()
                )


            if (
                aggregation_result.get(
                    "status"
                )
                != "aggregation completed"
            ):
                raise RuntimeError(
                    "Aggregation did not "
                    "complete successfully: "
                    f"{aggregation_result}"
                )
            
            


            print(
                "Accepted clients:",
                aggregation_result[
                    "accepted_clients"
                ],
            )

            print(
                "Trust scores:",
                aggregation_result[
                    "trust_scores"
                ],
            )


            model_relative_change = (
                aggregation_result[
                    "model_relative_change"
                ]
            )


            print(
                "Model relative change:",
                model_relative_change,
            )


            
            # Global validation evaluation
            

            val_accuracy = None
            val_loss = None
            val_loss_change = None


            if (
                model_relative_change
                <
                MODEL_CHANGE_THRESHOLD
            ):

                response = requests.get(
                    f"{SERVER}/evaluate"
                )

                response.raise_for_status()

                evaluation = (
                    response.json()
                )


                val_accuracy = evaluation[
                    "accuracy"
                ]

                val_loss = evaluation[
                    "loss"
                ]


                if (
                    previous_val_loss
                    is not None
                ):

                    val_loss_change = abs(
                        val_loss
                        -
                        previous_val_loss
                    )


                    print(
                        "Validation loss change:",
                        val_loss_change,
                    )


                    if (
                        val_loss_change
                        <
                        VAL_LOSS_CHANGE_THRESHOLD
                    ):
                        stable_checks += 1

                    else:
                        stable_checks = 0


                previous_val_loss = val_loss


                print(
                    f"Stable checks: "
                    f"{stable_checks}/"
                    f"{PATIENCE}"
                )


            else:

                stable_checks = 0
                previous_val_loss = None

                print(
                    "Validation evaluation "
                    "skipped."
                )


            round_converged = (
                stable_checks
                >=
                PATIENCE
            )


            
            # Malicious-client statistics
            

            selected_malicious = sorted(
                set(
                    main_ids
                    +
                    backup_ids
                )
                &
                malicious_ids
            )


            accepted_ids = []


            for cluster_clients in (
                aggregation_result[
                    "accepted_clients"
                ].values()
            ):

                for item in cluster_clients:

                    if isinstance(
                        item,
                        (list, tuple),
                    ):
                        client_id = item[0]

                    else:
                        client_id = item


                    accepted_ids.append(
                        int(client_id)
                    )


            malicious_kept = [
                client_id
                for client_id
                in selected_malicious
                if client_id
                in accepted_ids
            ]


            malicious_rejected = [
                client_id
                for client_id
                in selected_malicious
                if client_id
                not in accepted_ids
            ]


            
            # Save round CSV
            

            save_round_to_csv(
                "results/proposed.csv",
                {
                    "round":
                        round_id,

                    "accuracy":
                        val_accuracy,

                    "loss":
                        val_loss,

                    "relative_change":
                        model_relative_change,

                    "representation_fairness":
                        representation_fairness,

                    "hellinger_distance":
                        hellinger_distance,

                    "selected_malicious":
                        len(
                            selected_malicious
                        ),

                    "malicious_kept":
                        len(
                            malicious_kept
                        ),

                    "malicious_rejected":
                        len(
                            malicious_rejected
                        ),

                },
            )


            
            # Print evaluation
            

            if (
                val_accuracy
                is not None
            ):

                print(
                    f"Global Accuracy: "
                    f"{val_accuracy:.4f}"
                )

                print(
                    f"Global Loss: "
                    f"{val_loss:.6f}"
                )

            else:

                print(
                    "Global Accuracy: "
                    "SKIPPED"
                )

                print(
                    "Global Loss: "
                    "SKIPPED"
                )


            finish_remaining_backups(
                backup_futures,
                round_id,
            )

            # Stopping criterion
            if round_converged:

                print(
                    "\nTraining stopped:"
                )

                print(
                    "Convergence criteria "
                    "satisfied."
                )

                converged = True

                break


            print(
                f"ROUND {round_id} completed."
            )


    
    # Final evaluation on held-out TEST set

    if converged:
        print(
            "\nTraining finished by convergence."
        )
    else:
        print(
            "\nTraining finished after "
            "reaching the maximum number of rounds."
        )

    print(
        "\n========================================"
    )
    print(
        "FINAL EVALUATION ON TEST SET"
    )
    print(
        "========================================"
    )

    response = requests.get(
        f"{SERVER}/evaluate_final"
    )

    response.raise_for_status()

    test_evaluation = response.json()

    test_accuracy = (
        test_evaluation["accuracy"]
    )

    test_loss = (
        test_evaluation["loss"]
    )

    test_macro_f1 = (
        test_evaluation.get(
            "macro_f1"
        )
    )

    test_balanced_accuracy = (
        test_evaluation.get(
            "balanced_accuracy"
        )
    )

    test_worst_class_accuracy = (
        test_evaluation.get(
            "worst_class_accuracy"
        )
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Loss: "
        f"{test_loss:.6f}"
    )

    if test_macro_f1 is not None:
        print(
            f"Macro F1: "
            f"{test_macro_f1:.4f}"
        )

    if test_balanced_accuracy is not None:
        print(
            f"Balanced Accuracy: "
            f"{test_balanced_accuracy:.4f}"
        )

    if test_worst_class_accuracy is not None:
        print(
            f"Worst-class Accuracy: "
            f"{test_worst_class_accuracy:.4f}"
        )

    # save_round_to_csv(
    #     "results/heterogeneity.csv",
    #     {
    #         "dirichlet_alpha":
    #             DIRICHLET_ALPHA,
    #
    #         "test_accuracy":
    #             test_accuracy,
    #
    #         "test_loss":
    #             test_loss,
    #
    #         "macro_f1":
    #             test_macro_f1,
    #
    #         "balanced_accuracy":
    #             test_balanced_accuracy,
    #
    #         "worst_class_accuracy":
    #             test_worst_class_accuracy,
    #     },
    # )
# Required for ProcessPoolExecutor


if __name__ == "__main__":
    main()