# active_learning/uncertainty_sampling.py

def compute_uncertainty(predictions):
    uncertainties = []
    for pred in predictions:
        if pred:
            max_confidence = max([box['confidence'] for box in pred])
            uncertainty = 1 - max_confidence
        else:
            uncertainty = 1.0  # No detections imply high uncertainty
        uncertainties.append(uncertainty)
    return uncertainties
