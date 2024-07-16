import torch
from torch.nn import functional as F


def relu_evidence(y):
    return F.relu(y)


def softplus_evidence(y):
    return F.softplus(y)


def exptanh_evidence(y, tao):
    return torch.exp(torch.tanh(y) / tao)


def sigmoid_evidence(y, tao):
    return (F.sigmoid(y) / tao)


def one_hot_embedded(n_classes, input_tensor):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def kl_divergence(alpha, num_classes, batch):
    # ones = torch.ones([batch, num_classes, 384, 384], dtype=torch.float32).cuda()
    # ones = torch.ones([batch, num_classes, 256, 256], dtype=torch.float32).cuda()
    ones = torch.ones_like(alpha)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),                      # 0.5
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    b = y.shape[0]
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, batch=b)
    return A + kl_div


def edl_loss_un(func, y, alpha, epoch_num, num_classes, annealing_step, mask):
    S = torch.sum(alpha, dim=1, keepdim=True)
    Y = mask * y
    A = torch.sum(Y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(0.5, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (mask - Y) + 1
    b = y.shape[0]
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, batch=b)
    return A + kl_div


def edl_log_loss(evidence, target, epoch_num, num_classes, annealing_step):
    # evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step
        )
    )
    return loss


def edl_digamma_loss(evidence, target, epoch_num, num_classes, annealing_step):
    # evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step)
    )
    return loss


def edl_digamma_loss_un(evidence, target, epoch_num, num_classes, annealing_step, mask):
    # evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss_un(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, mask
        )
    )
    return loss


def edl_mse_loss(evidence, target, epoch_num, num_classes, annealing_step):
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step)
    )
    return loss


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step):
    loglikelihood = loglikelihood_loss(y, alpha)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return loglikelihood + kl_div


def loglikelihood_loss(y, alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def digamma_loss(alpha, target, epoch_num, num_classes, annealing_step):
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step
        )
    )

    return loss


def d_loss(y, alpha, epoch_num, num_classes, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    b = y.shape[0]
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, batch=b)
    return A + kl_div