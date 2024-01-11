In this branch (spookylike_training_new_opt) I try adapting the learning rate by creating a new optax gradient transformation
that uses the same opt_state as before (because the optstate has an 'EmptyState' attribute in the attribute which corresponds
to the scale_by_learning_rate transformation).

