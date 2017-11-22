import tensorflow as tf
import numpy as np

class SigmoidFocalClassificationLoss():
  """Sigmoid focal cross entropy loss.
  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """

  def __init__(self, anchorwise_output=False, gamma=2.0, alpha=0.25):
    """Constructor.
    Args:
      anchorwise_output: Outputs loss per anchor. (default False)
      gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
      alpha: optional alpha weighting factor to balance positives vs negatives.
    """

    self._anchorwise_output = anchorwise_output
    self._alpha = alpha
    self._gamma = gamma

  def compute_loss(self,
                    prediction_tensor,
                    target_tensor,
                    weights,
                    class_indices=None):
    """Compute loss function.
    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.
    Returns:
      loss: a (scalar) tensor representing the value of the loss function
            or a float tensor of shape [batch_size, num_anchors]
    """
    weights = tf.expand_dims(weights, 2)
    if class_indices is not None:
      weights *= tf.reshape(
          ops.indices_to_dense_vector(class_indices,
                                      tf.shape(prediction_tensor)[2]),
          [1, 1, -1])
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    prediction_probabilities = tf.sigmoid(prediction_tensor)
    p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if self._gamma:
      modulating_factor = tf.pow(1.0 - p_t, self._gamma)
    alpha_weight_factor = 1.0
    if self._alpha is not None:
      alpha_weight_factor = (target_tensor * self._alpha +
                             (1 - target_tensor) * (1 - self._alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)
    if self._anchorwise_output:
    #   return tf.reduce_sum(focal_cross_entropy_loss * weights, 2)
    # return tf.reduce_sum(focal_cross_entropy_loss * weights)    
      return tf.reduce_mean(focal_cross_entropy_loss * weights, 2)
    return tf.reduce_mean(focal_cross_entropy_loss * weights)