import math

def evaluate_shadow(production_log: list, shadow_log: list, criteria: dict) -> dict:
    """
    Returns a dictionary with the promotion decision and metrics.
    """

    size = len(production_log)
    latency_shad = []
    prod_acc = 0
    shad_acc = 0
    agg_rate = 0
    
    
    for log_index in range(size):
        prod_log = production_log[log_index]
        shad_log = shadow_log[log_index]

        prod_acc += (prod_log['prediction'] == prod_log['actual'])
        shad_acc += (shad_log['prediction'] == shad_log['actual'])

        agg_rate += (shad_log['prediction'] == prod_log['prediction'])

        latency_shad.append(shad_log['latency_ms'])

    prod_acc /= size
    shad_acc /= size
    acc_gain = shad_acc - prod_acc
    agg_rate /= size
    latency_shad.sort()

    p95_index = math.ceil(0.95*size)-1
    p95 = latency_shad[p95_index]

    promote_flag = ( acc_gain >= criteria['min_accuracy_gain']) & ( p95 <= criteria['max_latency_p95']) &  ( agg_rate >= criteria['min_agreement_rate']) 

    result  = {
        "promote" : promote_flag,
        "metrics" :  {"shadow_accuracy": shad_acc, "production_accuracy": prod_acc, "accuracy_gain": acc_gain, "shadow_latency_p95": p95, "agreement_rate": agg_rate}

        
    }

    return result
        
        
        