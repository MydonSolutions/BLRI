
def hdf5_field_get(h5_field):
    if h5_field.shape == tuple():
        return h5_field[()]
    else:
        return h5_field[:]


def hdf5_fields_are_equal(
        h5_a_field,
        h5_b_field,
    ):
    if h5_a_field.shape == tuple():
        return h5_a_field[()] == h5_b_field[()]
    else:
        if h5_a_field.shape == h5_b_field.shape:
            return (h5_a_field[:] == h5_b_field[:]).all()
        else:
            return False