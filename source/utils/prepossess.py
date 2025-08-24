import torch
import numpy as np

def append_to_4head(pearson_matrices: torch.Tensor):
    pass
    

def pearson_to_vector(pearson_matrices: torch.Tensor) -> torch.Tensor:
    """Converts a batch of Pearson matrices to a batch of vectors.
    
    Each vector contains the flattened upper triangle of a matrix.
    This is the required input format for the MLP model.
    """
    # This check ensures we don't try to flatten an already flat vector
    if pearson_matrices.ndim != 3:
        return pearson_matrices

    batch_size, n, _ = pearson_matrices.shape
    # Get indices of the upper triangle, k=1 means excluding the diagonal
    triu_indices = torch.triu_indices(n, n, offset=1, device=pearson_matrices.device)
    
    # Apply indices to each matrix in the batch
    vectors = pearson_matrices[:, triu_indices[0], triu_indices[1]]
    return vectors

def continus_mixup_data(*tensors, alpha=1.0):
    """
    Performs Mixup augmentation on a list of tensors.
    The last tensor in the list is assumed to be the label.

    Args:
        *tensors: A sequence of tensors to be mixed. The last tensor must be the label.
                  All tensors must be on the same device and have the same batch size.
        alpha (float): The hyperparameter for the Beta distribution.

    Returns:
        A tuple of mixed tensors, with the same length and order as the input.
    """
    # 1. Separate the input tensors from the label tensor
    *xs, y = tensors
    
    # 2. Robustly determine the device from the input tensors
    #    This eliminates the hardcoded 'cuda'
    if not xs:
        # If only labels are passed, no mixing is possible
        return (*xs, y)
    device = xs[0].device
    
    # 3. Generate the mixing coefficient
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # 4. Generate the shuffled indices for mixing
    batch_size = y.size(0)
    if batch_size <= 1:
        # Cannot mix if batch size is 1 or less
        return (*xs, y)
        
    index = torch.randperm(batch_size, device=device)

    # 5. Mix the input feature tensors
    mixed_xs = []
    for x in xs:
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_xs.append(mixed_x)
    
    # 6. Mix the label tensor
    mixed_y = lam * y + (1 - lam) * y[index]
    
    # 7. Return the results as a single tuple, ready for unpacking
    #    The star operator (*) unpacks the list into the tuple
    return (*mixed_xs, mixed_y)



def mixup_data_by_class(*tensors, alpha=1.0):
    """
    Performs intra-class Mixup. It only mixes samples that belong to the same class.

    Args:
        *tensors: A sequence of tensors. The last tensor must be the label.
                  The label tensor should be one-hot encoded for this to work correctly.

    Returns:
        A tuple of mixed tensors, with shuffled but corresponding samples.
    """
    # 1. Separate features from the one-hot encoded label
    *xs, y = tensors
    device = xs[0].device
    
    # We need the class indices, not the one-hot vector, to group by class
    #class_indices = torch.argmax(y, dim=1)
    
    # Lists to store the mixed results from each class
    final_mixed_xs = [[] for _ in xs]
    final_mixed_ys = []

    # 2. Iterate through each unique class present in the batch
    for class_id in torch.unique(y):
        # Find the batch indices corresponding to the current class
        mask = (y == class_id)
        
        # Select the data for the current class
        class_xs = [x[mask] for x in xs]
        class_y = y[mask]
        
        # 3. Perform standard Mixup ONLY on the data for this class
        #    We construct a new tuple to pass to continus_mixup_data
        tensors_for_mixup = (*class_xs, class_y)
        mixed_results = continus_mixup_data(*tensors_for_mixup, alpha=alpha)
        
        # 4. Separate the results and store them
        *mixed_class_xs, mixed_class_y = mixed_results
        
        for i in range(len(xs)):
            final_mixed_xs[i].append(mixed_class_xs[i])
        
        final_mixed_ys.append(mixed_class_y)
    
    # 5. Concatenate the mixed batches from all classes
    #    This will shuffle the original order of the samples, but that's
    #    fine for training as long as features and labels correspond.
    cat_xs = [torch.cat(x_list, dim=0) for x_list in final_mixed_xs]
    cat_y = torch.cat(final_mixed_ys, dim=0)
    
    return (*cat_xs, cat_y)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_cluster_loss(matrixs, y, intra_weight=2):

    y_1 = y[:, 1]

    y_0 = y[:, 0]

    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = 0.0

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss


def inner_loss(label, matrixs):

    loss = 0

    if torch.sum(label == 0) > 1:
        loss += torch.mean(torch.var(matrixs[label == 0], dim=0))

    if torch.sum(label == 1) > 1:
        loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

    return loss


def intra_loss(label, matrixs):
    a, b = None, None

    if torch.sum(label == 0) > 0:
        a = torch.mean(matrixs[label == 0], dim=0)

    if torch.sum(label == 1) > 0:
        b = torch.mean(matrixs[label == 1], dim=0)
    if a is not None and b is not None:
        return 1 - torch.mean(torch.pow(a-b, 2))
    else:
        return 0
