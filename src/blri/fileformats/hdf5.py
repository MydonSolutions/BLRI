
def hdf5_fields_are_equal(
        h5_a_field,
        h5_b_field,
    ):
    try:
        if h5_a_field.shape == tuple():
            return h5_a_field[()] == h5_b_field[()]
        else:
            return (h5_a_field[:] == h5_b_field[:]).all()
    except ValueError:
        return False