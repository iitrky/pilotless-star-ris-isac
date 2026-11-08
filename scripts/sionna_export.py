"""Ray-trace the STAR-RIS link segments with Sionna RT and save per-slot channel snapshots.

Written against the Sionna RT >= 1.0 API (load_scene / PathSolver / Paths.cfr). Sionna's RT API
changed between 0.x and 1.x: check names against your installed version before running.

STAR-RIS handling: every surface element is modelled as an isotropic point receiver (for the
BS->RIS segment) and point transmitter (for RIS->vehicle/target/sensor segments). The cascaded
channels are formed in post-processing with the agent's Theta_r / Theta_t (paper Sec. 6.1), so no
native STAR-RIS support is required.

Usage:
  python scripts/sionna_export.py --scene simple_street_canyon --traj traj.csv --out rt_canyon.npz
traj.csv columns: t, id, x, y, z, vx, vy, kind   (kind: vehicle | target), e.g. from the MATLAB
script matlab/make_trajectories.m
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="simple_street_canyon")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--fc", type=float, default=7e9)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--Nx", type=int, default=8)
    ap.add_argument("--Nz", type=int, default=8)
    ap.add_argument("--ris", default="0,0,8")
    ap.add_argument("--bs", default="-15,20,12")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--out", default="rt_snapshots.npz")
    args = ap.parse_args()

    import sionna.rt as rt
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

    scene = load_scene(getattr(rt.scene, args.scene))
    scene.frequency = args.fc
    lam = 3e8 / args.fc
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    solver = PathSolver()

    ris_c = np.array([float(v) for v in args.ris.split(",")])
    bs = np.array([float(v) for v in args.bs.split(",")])
    ix = (np.arange(args.Nx) - (args.Nx - 1) / 2) * lam / 2
    iz = (np.arange(args.Nz) - (args.Nz - 1) / 2) * lam / 2
    elems = np.array([[ris_c[0] + x, ris_c[1], ris_c[2] + z] for x in ix for z in iz])
    bs_ant = np.array([[bs[0] + m * lam / 2, bs[1], bs[2]] for m in range(args.M)])

    def link(tx_pos, rx_pos):
        """Narrowband channel matrix (len(rx), len(tx)) between point sets at fc."""
        for n in list(scene.transmitters):
            scene.remove(n)
        for n in list(scene.receivers):
            scene.remove(n)
        for i, p in enumerate(tx_pos):
            scene.add(Transmitter(name=f"tx{i}", position=[float(v) for v in p]))
        for j, p in enumerate(rx_pos):
            scene.add(Receiver(name=f"rx{j}", position=[float(v) for v in p]))
        paths = solver(scene, max_depth=args.max_depth, los=True, specular_reflection=True,
                       diffuse_reflection=False, refraction=True, synthetic_array=False)
        h = paths.cfr(frequencies=np.array([0.0]), normalize=False, out_type="numpy")
        # h shape: [num_rx, rx_ant, num_tx, tx_ant, num_time_steps, num_freq]
        return np.asarray(h)[:, 0, :, 0, 0, 0]

    G = link(bs_ant, elems)                                   # (N, M), static
    tr = np.genfromtxt(args.traj, delimiter=",", names=True, dtype=None, encoding=None)
    times = np.unique(tr["t"])
    g_list, hd_list, tg_list, pv_list = [], [], [], []
    for t in times:
        rows = tr[tr["t"] == t]
        veh = rows[rows["kind"] == "vehicle"]
        pv = np.stack([veh["x"], veh["y"], veh["z"]], 1)
        pv_list.append(np.concatenate([pv, np.stack([veh["vx"], veh["vy"]], 1)], 1))
        g_list.append(link(elems, pv))                        # (K, N): row k = g_k^H (before Theta)
        hd_list.append(link(bs_ant, pv))                      # (K, M)
        tgt = rows[rows["kind"] == "target"]
        tg_list.append(np.stack([tgt["x"], tgt["y"], tgt["z"]], 1))
        print(f"t={t:.4f}s done")
    np.savez_compressed(args.out, G=G, g=np.stack(g_list), hd=np.stack(hd_list), p_tgt=np.stack(tg_list), p_veh=np.stack(pv_list),
                        times=times, elems=elems, bs_ant=bs_ant)


if __name__ == "__main__":
    main()
