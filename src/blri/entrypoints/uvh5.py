import argparse

from blri.fileformats.uvh5 import h5py, uvh5_differences

def diff():
    parser = argparse.ArgumentParser(
        description="Compare 2 UVH5 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "uvh5_filepaths",
        type=str,
        nargs=2,
        help="The paths to the 2 UVH5 files to compare.",
    )
    parser.add_argument(
        "-a", "--atol",
        type=float,
        default=1e-8,
        help="The absolute tolerance in visdata comparison (see `numpy.isclose`).",
    )
    parser.add_argument(
        "-r", "--rtol",
        type=float,
        default=1e-5,
        help="The absolute tolerance in visdata comparison (see `numpy.isclose`).",
    )
    parser.add_argument(
        "-H", "--header-exclude",
        type=str,
        nargs="+",
        default=["history"],
        help="/Header/* datasets to exclude in the comparison.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase the verbosity of the comparison (0=Silent, 1=Print Field Names, 2=Print Fields (Header only), 3=Print Fields)."
    )
    args = parser.parse_args()

    header_fields_diff, data_fields_diff = uvh5_differences(
        *args.uvh5_filepaths,
        atol=args.atol, rtol=args.rtol
    )

    header_fields_diff = set(header_fields_diff).difference(
        set(args.header_exclude)
    )
    if args.verbose == 0:
        if len(header_fields_diff) + len(data_fields_diff) > 0:
            exit(1)
        exit(0)
    
    h5files = None
    if args.verbose > 1:
        h5files = {
            filepath: h5py.File(filepath, 'r')
            for filepath in args.uvh5_filepaths
        }

    for group, fields_diff in {
        "Header": header_fields_diff,
        "Data": data_fields_diff,
    }.items():
        for field in fields_diff:
            print(f"/{group}/{field}")
            if args.verbose == 1:
                continue

            if args.verbose == 2 and group == "Data":
                continue
            
            fields = {
                filepath: h5file[group][field]
                for filepath, h5file in h5files.items()
            }
            field_shapes = list(set([
                field.shape
                for field in fields.values()
            ]))
            if len(field_shapes) != 1:
                print(f"  Field shape mismatch: {field_shapes}")
            else:
                for filepath, field_ in fields.items():
                    if field_shapes[0] == tuple():
                        print(f"  {filepath}:\n    {field_[()]}")
                    else:
                        print(f"  {filepath}: {field_shapes[0]}\n{field_[:]}")

    if h5files is not None:
        for h5file in h5files.values():
            h5file.close()
