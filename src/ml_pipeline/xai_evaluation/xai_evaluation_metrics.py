import numpy as np
import shap
from scipy import stats


### utils ###
def get_explanation(x, explainer):
    """ wrapper to get explanations from lime /shap """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    if isinstance(explainer, shap.explainers.Tree):
        exp = explainer.shap_values(x)
    elif isinstance(explainer, shap.explainers.other.LimeTabular):
        exp = explainer.attributions(x)
    elif isinstance(explainer, shap.explainers.other.Random):
        exp = explainer(x).values
    else:
        raise ValueError("explainer unknown")
    return exp

### Faithfulness ###
def faithfulness_metric(model, x, y_true, coefs, base):
    """ This metric evaluates the correlation between the importance assigned by the interpretability algorithm
    to attributes and the effect of each of the attributes on the performance of the predictive model.
    The higher the importance, the higher should be the effect, and vice versa, The metric evaluates this by
    incrementally removing each of the attributes deemed important by the interpretability metric, and
    evaluating the effect on the performance, and then calculating the correlation between the weights (importance)
    of the attributes and corresponding model performance. [#]_

    Adapted to regression case:
        Effect on classifier is measured as:
            1. absolute error of modified prediction with original prediction
            2. absolute error of modified predcition with true value


    This function was adapted from the implementation: https://github.com/Trusted-AI/AIX360/blob/master/aix360/metrics/local_metrics.py

    References:
        .. [#] `David Alvarez Melis and Tommi Jaakkola. Towards robust interpretability with self-explaining
           neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors,
           Advances in Neural Information Processing Systems 31, pages 7775-7784. 2018.
           <https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf>`_
    Args:
        model: Trained classifier, such as a ScikitClassifier that implements
            a predict() and a predict_proba() methods.
        x (numpy.ndarray): row of data.
        coefs (numpy.ndarray): coefficients (weights) corresponding to attribute importance.
        base ((numpy.ndarray): base (default) values of attributes
    Returns:
        float: correlation between attribute importance weights and corresponding effect on classifier.
    """
    assert coefs.shape == (x.shape[1],)

    # prediction of original sample
    pred = model.predict(x)[0]

    # take abs of coefs
    coefs = np.abs(coefs)

    # find indexs of coefficients in decreasing order of value
    ar = np.argsort(-coefs)  # argsort returns indexes of values sorted in increasing order; so do it for negated array
    preds = np.zeros(x.shape[1])
    pr_diffs = np.zeros(x.shape[1])
    true_diffs = np.zeros(x.shape[1])
    for ind in np.nditer(ar):
        # print(ind)
        x_copy = x.copy()
        # print(x.copy())
        # remove feature by replacing with baseline value
        x_copy[0][ind] = base[ind]
        # predict with modified sample
        x_copy_pred = model.predict(x_copy)[0]
        preds[ind] = x_copy_pred
        # calculate deviation of modified prediction with original prediction → performance metric
        pr_diff = np.abs(pred - x_copy_pred)
        pr_diffs[ind] = pr_diff
        # true diff
        true_diff = np.abs(y_true - x_copy_pred)
        true_diffs[ind] = true_diff

    # corr: coefs with (pred - modified pred)
    pred_cor = np.corrcoef(coefs, pr_diffs)[0, 1]

    # corr: coefs with (y_true - modified pred)
    true_cor = np.corrcoef(coefs, true_diffs)[0, 1]

    return pred_cor, true_cor  # , coefs, preds, pr_diffs, true_diffs, ar


def eval_faithfulness(x, y, attributions, model, base_values):
    """Calculate faithfulness for whole dataset

    Args:
        x (np.array): input data
        y (np.array): target data
        attributions (np.array): attribution weights
        model: Trained classifier, such as a ScikitClassifier that implements
            a predict() and a predict_proba() methods.
        base_values: baseline values used to `remove` features
    """
    assert x.shape[0] == y.shape[0] == attributions.shape[0]
    assert x.shape[1] == attributions.shape[1] == base_values.shape[0]

    pred_cors = []
    true_cors = []

    for sample_ind, x_sample in enumerate(x):
        x_sample = x_sample.reshape(1, -1)
        y_true = y[sample_ind]
        attribution_arr = attributions[sample_ind]
        pred_cor, true_cor = faithfulness_metric(model, x_sample, y_true, attribution_arr, base_values)
        pred_cors.append(pred_cor)
        true_cors.append(true_cor)

    print("Pred corr")
    print(stats.describe(pred_cors))
    print("True corr")
    print(stats.describe(true_cors))

    return pred_cors #, true_cors

### Robustness / Stability
# below implementations are adapted from https://github.com/AI4LIFE-GROUP/OpenXAI/

class MarginalPerturbation():
    def __init__(self, dist_std_per_feature):
        '''
        Initializes the marginal perturbation method where each column is sampled from marginal distributions given per variable.
        dist_per_feature : vector of distribution generators (tdist under torch.distributions).
        Note : These distributions are assumed to have zero mean since they get added to the original sample.
        '''
        self.dist_std_per_feature = dist_std_per_feature

    def get_perturbed_inputs(self, original_sample: np.array, feature_mask: np.array, num_samples: int) -> np.array:
        '''
        feature mask : this indicates features to perturb
        num_samples : number of perturbed samples.
        '''
        perturbed_cols = []
        for i, _ in enumerate(original_sample):
            perturbed_cols.append(np.random.normal(0, self.dist_std_per_feature[i], num_samples).reshape(-1, 1))
        perturbed_samples = original_sample + np.concatenate(perturbed_cols, 1) * (feature_mask)

        # return self._filter_out_of_range_samples(original_sample, perturbed_samples, max_distance)
        return perturbed_samples

def compute_Lp_norm_diff(vec1, vec2, normalize_to_relative_change: bool = True, pnorm: int = 2):
    """ Returns the Lp norm of the difference between vec1 and vec2.
    Args:
        normalize_to_relative_change: when true, normalizes the difference between vec1 and vec2 by vec1
    """

    # arrays can be flattened, so long as ordering is preserved
    flat_diff = vec1.flatten() - vec2.flatten()
    if normalize_to_relative_change:
        vec1_arr = vec1.flatten()
        flat_diff = np.divide(flat_diff, vec1_arr, where=vec1_arr != 0)
    return np.linalg.norm(flat_diff, ord=pnorm)


def evaluate_stability_metric(x, model, explainer, perturber, stability_metric, feature_mask,
                              num_samples=100, max_pred_diff=0.005, eps=0.0001, random_seed=100):
    """

    Args:
        x:
        model:
        explainer:
        perturber:
        feature_mask: feature to perturb
        stability_metric: 'ROS', 'RIS'
        num_samples:
        max_pred_diff: maximum difference in prediction to still comply to y_pred == y_pred'
        eps: norm in demominator is clipped by eps to avoid division by 0

    Returns:

    """
    np.random.seed(random_seed)

    if len(x.shape) == 1:
        original_sample = x.reshape(1, -1)
    else:
        original_sample = x

    # original prediction / explanation
    y_pred = model.predict(original_sample)
    exp_original = get_explanation(original_sample, explainer)

    # get perturbed samples
    x_prime_samples = perturber.get_perturbed_inputs(original_sample, feature_mask=feature_mask,
                                                     num_samples=num_samples)
    y_prime_preds = model.predict(x_prime_samples)

    # select perturbed samples that have same prediction as original sample
    y_pred_diffs = np.abs(y_prime_preds - y_pred)
    ind_equal_y_pred = y_pred_diffs < max_pred_diff
    # print(f"select {np.sum(ind_equal_y_pred)}/{num_samples} samples")
    x_prime_samples = x_prime_samples[ind_equal_y_pred]
    y_prime_preds = y_prime_preds[ind_equal_y_pred]

    # calculate the explanation for each perturbation sample
    exp_prime_samples = np.zeros_like(x_prime_samples)
    for it, x_prime in enumerate(x_prime_samples):
        x_prime = x_prime.reshape(1, -1)
        y_prime_pred = y_prime_preds[it]
        exp = get_explanation(x_prime, explainer)
        exp_prime_samples[it, :] = exp

    # calculate stability ratio for each perturbation sample

    stability_ratios = []  # record ratio per sample
    x_diffs = []
    y_diffs = []
    exp_diffs = []
    for sample_ind, x_prime in enumerate(x_prime_samples):

        # explanation diff
        exp_at_perturbation = exp_prime_samples[sample_ind]
        explanation_diff = compute_Lp_norm_diff(exp_original, exp_prime_samples[sample_ind],
                                                normalize_to_relative_change=True)
        exp_diffs.append(explanation_diff)

        if stability_metric == "RIS":
            x_diff = compute_Lp_norm_diff(original_sample, x_prime, normalize_to_relative_change=True)
            x_diff = np.max((x_diff, eps))
            x_diffs.append(x_diff)
            stability_ratio = np.divide(explanation_diff, x_diff)
        elif stability_metric == "ROS":
            y_diff = compute_Lp_norm_diff(y_pred, y_prime_preds[sample_ind], normalize_to_relative_change=False)
            #if y_diff <= eps: print("small y_diff:", y_diff)
            y_diff = np.max((y_diff, eps))
            y_diffs.append(y_diff)
            stability_ratio = np.divide(explanation_diff, y_diff)
        else:
            raise ValueError("stability metric unknown")

        stability_ratios.append(stability_ratio)

    # select max stability ratio
    ind_max = np.argmax(stability_ratios)

    return stability_ratios[ind_max]  # , stability_ratios, y_diffs, x_diffs, exp_diffs, ind_max


