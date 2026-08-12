
def update_trust_score(old_trust, A_i, lambda_trust):

    new_trust = (lambda_trust*old_trust + (1 - lambda_trust)*(1 / (1 + A_i)))
    return new_trust

