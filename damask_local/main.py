import difflib
import inspect
import requests
import subprocess
import yaml
import warnings
import numpy as np
from functools import cache, wraps
from hashlib import sha256
from pathlib import Path

from damask import YAML, ConfigMaterial, Rotation, GeomGrid, seeds, Result
from mendeleev.fetch import fetch_table


# Sentinel wrapper
class ExplicitDefault:
    def __init__(self, default, msg=None):
        self.default = default
        self.msg = msg


def use_default(default, msg=None):
    return ExplicitDefault(default, msg)


# Decorator
def with_explicit_defaults(func):
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        # Track updated arguments
        new_args = list(args)
        new_kwargs = dict(kwargs)

        for name, param in sig.parameters.items():
            default_val = param.default

            if not isinstance(default_val, ExplicitDefault):
                continue  # Only handle sentinel-wrapped defaults

            # Determine if argument was passed
            was_explicit = (
                name in bound.arguments
                and name in kwargs
                or (
                    param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD]
                    and list(sig.parameters).index(name) < len(args)
                )
            )

            if not was_explicit:
                # Not passed: Replace sentinel with real default
                if default_val.msg:
                    warnings.warn(default_val.msg)
                else:
                    warnings.warn(
                        f"'{name}' not provided, using default: {default_val.default}"
                    )
                if param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD]:
                    idx = list(sig.parameters).index(name)
                    if idx < len(new_args):
                        new_args[idx] = default_val.default
                    else:
                        new_kwargs[name] = default_val.default
                else:
                    new_kwargs[name] = default_val.default

        return func(*new_args, **new_kwargs)

    return wrapper


@cache
def get_metadata(key):
    # define the path to the metadata file relative to this file:
    path = Path(__file__).parent / "data" / "metadata.yml"
    with open(path, "r") as file:
        metadata = yaml.safe_load(file)[key]
    return metadata


def look_up_name(chemical_composition: list[str], key: str):
    metadata = get_metadata(key)
    all_data = [
        data
        for data in metadata
        if all(c in data.get("composition", {}) for c in chemical_composition)
    ]
    if len(all_data) == 0:
        raise ValueError(
            f"No data found for the given chemical composition: {chemical_composition}"
        )
    elif len(all_data) == 1:
        return [all_data[0]["name"]]
    return [
        all_data[ii]["name"]
        for ii in _order_composition(
            [data["composition"] for data in all_data], chemical_composition
        )
    ]


def _order_composition(
    composition: list[dict[str, str | float]],
    selected_elements: list[str],
) -> list[str]:
    all_values = []
    for comp in composition:
        value = 0
        for elem in selected_elements:
            if comp[elem] == "balance":
                balance_value = 100
                for v in comp.values():
                    if isinstance(v, float | int):
                        balance_value -= v
                value += balance_value
            elif isinstance(comp[elem], float | int):
                value += comp[elem]
        all_values.append(value)
    return np.argsort(all_values).tolist()[::-1]


@cache
def list_elasticity(
    chemical_composition: str | list[str] | None = None,
    sub_folder="elastic",
    repo_owner="damask-multiphysics",
    repo_name="DAMASK",
    directory_path="examples/config/phase/mechanical",
):
    """
    Fetches all the elasticity YAML files in the specified directory from the
    specified GitHub repository.

    Args:
        sub_folder (str): The subfolder within the directory to fetch the YAML
            files from.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        directory_path (str): The path to the directory containing the YAML
            files.

    Returns:
        dict: A dictionary containing the YAML content of each file in the directory
    """
    data = get_yaml(sub_folder, repo_owner, repo_name, directory_path)
    if chemical_composition is None:
        return data
    if isinstance(chemical_composition, str):
        chemical_composition = [chemical_composition]
    names = look_up_name(chemical_composition, "elasticity")
    return {name: data[name] for name in names if name in data}


@cache
def list_plasticity(
    chemical_composition: str | list[str] | None = None,
    sub_folder="plastic",
    repo_owner="damask-multiphysics",
    repo_name="DAMASK",
    directory_path="examples/config/phase/mechanical",
):
    """
    Fetches all the plasticity YAML files in the specified directory from the specified GitHub repository.

    Args:
        sub_folder (str): The subfolder within the directory to fetch the YAML files from.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        directory_path (str): The path to the directory containing the YAML files.

    Returns:
        dict: A dictionary containing the YAML content of each file in the directory
    """
    data = get_yaml(sub_folder, repo_owner, repo_name, directory_path)
    if chemical_composition is None:
        return data
    if isinstance(chemical_composition, str):
        chemical_composition = [chemical_composition]
    names = look_up_name(chemical_composition, "plasticity")
    return {name: data[name] for name in names if name in data}


def get_yaml(
    sub_folder="",
    repo_owner="damask-multiphysics",
    repo_name="DAMASK",
    directory_path="examples/config/phase/mechanical",
):
    """
    Fetches all the YAML files in the specified directory from the specified GitHub repository.

    Args:
        sub_folder (str): The subfolder within the directory to fetch the YAML files from.
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        directory_path (str): The path to the directory containing the YAML files.

    Returns:
        dict: A dictionary containing the YAML content of each file in the directory
    """

    # GitHub API URL to get the directory contents
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{directory_path}/{sub_folder}"

    # Dictionary to store YAML content
    yaml_dicts = {}

    # Fetch directory contents
    response = requests.get(api_url)

    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file["name"].endswith(".yaml"):
                # Get raw file URL
                raw_url = file["download_url"]

                # Download the file
                file_response = requests.get(raw_url)
                if file_response.status_code == 200:
                    try:
                        # Load the YAML content into a Python dictionary
                        yaml_content = yaml.safe_load(file_response.text)
                        yaml_dicts[file["name"].replace(".yaml", "")] = yaml_content
                    except yaml.YAMLError as e:
                        warnings.warn(f"Failed to load {file['name']}: {e}")
                else:
                    warnings.warn(
                        f"Failed to download {file['name']}: {file_response.status_code}"
                    )
    else:
        response.raise_for_status()
    return yaml_dicts


def _get_lattice_structure(key=None, lattice=None, chemical_symbol=None):
    if key is None and lattice is None and chemical_symbol is None:
        raise ValueError(
            "At least one of 'key', 'lattice', or 'chemical_symbol' must be provided."
        )
    if lattice is None and key is not None:
        for k in get_metadata("elasticity"):
            if k["name"] == key:
                lattice = k.get("lattice_structure", None)
                break
    if lattice is None:
        if len(chemical_symbol) > 2:
            lattice = get_atom_info(name=chemical_symbol)["lattice_structure"]
        else:
            lattice = get_atom_info(symbol=chemical_symbol)["lattice_structure"]
    lattice = {"BCC": "cI", "HEX": "hP", "FCC": "cF", "BCT": "tI", "DIA": "cF"}.get(
        lattice.upper(), lattice
    )
    return lattice


def get_phase(
    elasticity,
    plasticity=None,
    chemical_symbol=None,
    lattice=None,
    output_list=None,
):
    """
    Returns a dictionary describing the phases for damask.

    For the details of isotropic model, one can refer to:
    https://doi.org/10.1016/j.scriptamat.2017.09.047
    """
    key = None
    if "type" not in elasticity:
        for k, v in elasticity.items():
            key = k
            elasticity = v
            break
    if plasticity is not None and "type" not in plasticity:
        plasticity = list(plasticity.values())[0]
    assert "type" in elasticity, "Problem with the elasticity format"
    if output_list is None:
        if plasticity is None:
            output_list = ["F", "P", "F_e"]
        else:
            output_list = ["F", "P", "F_e", "F_p", "L_p", "O"]
    d = {
        "lattice": _get_lattice_structure(key, lattice, chemical_symbol),
        "mechanical": {"output": output_list, "elastic": elasticity},
    }
    if plasticity is not None:
        d["mechanical"]["plastic"] = plasticity
    return {sha256(str(d).encode("utf-8")).hexdigest(): d}


def get_atom_info(difflib_cutoff=0.8, **kwargs):
    """
    Get atomic information from the periodic table.

    Args:
        difflib_cutoff (float): Cutoff for difflib.get_close_matches
        **kwargs: Key-value pairs to search for

    Returns:
        dict: Atomic information
    """
    df = fetch_table("elements")
    if len(kwargs) == 0:
        raise ValueError("No arguments provided")
    for key, tag in kwargs.items():
        if difflib_cutoff < 1:
            key = get_tag(key, df.keys(), cutoff=difflib_cutoff)
            tag = get_tag(tag, df[key], cutoff=difflib_cutoff)
            if sum(df[key] == tag) == 0:
                raise KeyError(f"'{tag}' not found")
            df = df[df[key] == tag]
    return df.squeeze(axis=0).to_dict()


def get_tag(tag, arr, cutoff=0.8):
    results = difflib.get_close_matches(tag, arr, cutoff=cutoff)
    if len(results) == 0:
        raise KeyError(f"'{tag}' not found")
    return results[0]


@with_explicit_defaults
def get_rotation(method="from_random", shape=use_default(8)):
    """
    Args:
        method (damask.Rotation.*/str): Method of damask.Rotation class which
            based on the given arguments creates the Rotation object. If
            string is given, it looks for the method within `damask.Rotation`
            via `getattr`.
        shape (int): Shape of the rotation matrix. If `method` is `from_random`,
            it defines the number of random rotations to be created.

    Returns:
        damask.Rotation: A Rotation object
    """
    if isinstance(method, str):
        method = getattr(Rotation, method)
    return method(shape=shape)


def generate_material(rotation, elements, phase, homogenization):
    _config = ConfigMaterial(
        {"material": [], "phase": phase, "homogenization": homogenization}
    )
    if not isinstance(rotation, (list, tuple, np.ndarray)):
        rotation = [rotation]
    if not isinstance(rotation, (list, tuple, np.ndarray)):
        elements = [elements]
    for r, e in zip(rotation, elements):
        _config = _config.material_add(
            O=r, phase=e, homogenization=list(homogenization.keys())[0]
        )
    return _config


def generate_load_step(
    N,
    t,
    F=None,
    dot_F=None,
    P=None,
    dot_P=None,
    f_out=None,
    r=None,
    f_restart=None,
    estimate_rate=None,
):
    """
    Args:
        N (int): Number of increments
        t (float): Time of load step in seconds, i.e.
        F (numpy.ndarray): Deformation gradient at end of load step
        dot_F (numpy.ndarray): Rate of deformation gradient during load step
        P (numpy.ndarray): First Piola–Kirchhoff stress at end of load step
        dot_P (numpy.ndarray): Rate of first Piola–Kirchhoff stress during
            load step
        r (float): Scaling factor (default 1) in geometric time step series
        f_out (int): Output frequency of results, i.e. f_out=3 writes results
            every third increment
        f_restart (int): output frequency of restart information; e.g.
            f_restart=3 writes restart information every tenth increment
        estimate_rate (float): estimate field of deformation gradient
            fluctuations based on former load step (default) or assume to be
            homogeneous, i.e. no fluctuations

    Returns:
        dict: A dictionary of the load step

    You can find more information about the parameters in the damask documentation:
    https://damask-multiphysics.org/documentation/file_formats/grid_solver.html#load-case
    """
    result = {
        "boundary_conditions": {"mechanical": {}},
        "discretization": {"t": t, "N": N},
    }
    if r is not None:
        result["discretization"]["r"] = r
    if f_out is not None:
        result["f_out"] = f_out
    if f_restart is not None:
        result["f_restart"] = f_restart
    if estimate_rate is not None:
        result["estimate_rate"] = estimate_rate
    if F is None and dot_F is None and P is None and dot_P is None:
        raise ValueError("At least one of the tensors should be provided.")
    if F is not None:
        result["boundary_conditions"]["mechanical"]["F"] = F
    if dot_F is not None:
        result["boundary_conditions"]["mechanical"]["dot_F"] = dot_F
    if P is not None:
        result["boundary_conditions"]["mechanical"]["P"] = P
    if dot_P is not None:
        result["boundary_conditions"]["mechanical"]["dot_P"] = dot_P
    return result


def generate_grid_from_voronoi_tessellation(
    spatial_discretization, num_grains, box_size
):
    if isinstance(spatial_discretization, (int, float)):
        spatial_discretization = np.array(3 * [spatial_discretization])
    if isinstance(box_size, (int, float)):
        box_size = np.array(3 * [box_size])
    seed = seeds.from_random(box_size, num_grains)
    return GeomGrid.from_Voronoi_tessellation(spatial_discretization, box_size, seed)


def get_homogenization(method=None, parameters=None):
    """
    Returns damask homogenization as a dictionary.
    Args:
        method(str): homogenization method
        parameters(dict): the required parameters
    """
    if method is None:
        method = "SX"
    if parameters is None:
        parameters = {"N_constituents": 1, "mechanical": {"type": "pass"}}
    return {method: parameters}


def generate_loading_tensor(default="F"):
    """
    Returns the default boundary conditions for the damask loading tensor.

    Args:
        default (str): Default value of the tensor. It can be 'F', 'P', 'dot_F'
            or 'dot_P'.

    Returns:
        tuple: A tuple of two numpy arrays. The first array is the keys and the
            second array is the values.
    """
    assert default in ["F", "P", "dot_F", "dot_P"]
    if default == "F":
        return np.full((3, 3), "F").astype("<U5"), np.eye(3)
    else:
        return np.full((3, 3), default).astype("<U5"), np.zeros((3, 3))


def loading_tensor_to_dict(key, value):
    """
    Converts the damask loading tensor to a dictionary.

    Args:
        key (numpy.ndarray): Keys of the tensor
        value (numpy.ndarray): Values of the tensor

    Returns:
        dict: A dictionary of the tensor

    Example:
        key, value = generate_loading_tensor()
        loading_tensor_to_dict(key, value)

    Comments:

        `key` and `value` should be generated from
        `generate_loading_tensor()` and as the format below:

        (array([['F', 'F', 'F'],
                ['F', 'F', 'F'],
                ['F', 'F', 'F']], dtype='<U5'),
         array([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]]))

        where the first array is the keys and the second array is the values.
        The keys can be 'F', 'P', 'dot_F' or 'dot_P'. These keys correspond to:

        F: deformation gradient at end of load step
        dot_F: rate of deformation gradient during load step
        P: first Piola–Kirchhoff stress at end of load step
        dot_P: rate of first Piola–Kirchhoff stress during load step
    """
    result = {}
    for tag in ["F", "P", "dot_F", "dot_P"]:
        if tag in key:
            mat = np.full((3, 3), "x").astype(object)
            mat[key == tag] = value[key == tag]
            result[tag] = mat.tolist()
    return result


def get_material(rotation, phase, homogenization):
    if not isinstance(rotation, (list, tuple, np.ndarray)):
        rotation = [rotation]
    return generate_material(rotation, list(phase.keys()), phase, homogenization)


@with_explicit_defaults
def get_grid(
    num_grains, box_size=use_default(1.0e-5), spatial_discretization=use_default(16)
):
    return generate_grid_from_voronoi_tessellation(
        box_size=box_size,
        spatial_discretization=spatial_discretization,
        num_grains=num_grains,
    )


@with_explicit_defaults
def apply_tensile_strain(strain=use_default(1.0e-3), default=use_default("dot_F")):
    keys, values = generate_loading_tensor(default)
    values[0, 0] = strain
    keys[1, 1] = keys[2, 2] = "P"
    data = loading_tensor_to_dict(keys, values)
    load_step = [
        generate_load_step(N=40, t=10, f_out=4, **data),
        generate_load_step(N=20, t=20, f_out=4, **data),
    ]
    return get_loading(solver={"mechanical": "spectral_basic"}, load_steps=load_step)


def get_loading(solver, load_steps):
    if not isinstance(load_steps, list):
        load_steps = [load_steps]
    return YAML(solver=solver, loadstep=load_steps)


def save_loading(loading, path, file_name="loading.yaml"):
    loading.save(path / file_name)
    return file_name


def run_damask(material, loading, grid, path=None):
    if path is None:
        path = Path(
            "damask_"
            + sha256(f"{material}_{loading}_{grid}".encode("utf-8")).hexdigest()
        )
    path = Path(path)
    path.mkdir(exist_ok=True)
    material.save(path / "material.yaml")
    loading.save(path / "loading.yaml")
    grid.save(path / "damask")

    command = "DAMASK_grid -m material.yaml -l loading.yaml -g damask.vti".split()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=path
    )
    stdout, stderr = process.communicate()
    return process, stdout, stderr, path


def average(d):
    return np.average(list(d.values()), axis=1)


def get_results(path, file_name="damask_loading_material.hdf5"):
    results = Result(path / file_name)
    results.add_stress_Cauchy()
    results.add_strain()
    results.add_equivalent_Mises("sigma")
    results.add_equivalent_Mises("epsilon_V^0.0(F)")
    stress = average(results.get("sigma"))
    strain = average(results.get("epsilon_V^0.0(F)"))
    stress_von_Mises = average(results.get("sigma_vM"))
    strain_von_Mises = average(results.get("epsilon_V^0.0(F)_vM"))
    return stress, strain, stress_von_Mises, strain_von_Mises
