def first_term_equal_zero(self):
    """
    Add the RLT constraints to the task.
    """
    self.handler.Constraints.first_term_equal_zero(
        num_matrices=self.handler.num_matrices_variables()
    )
