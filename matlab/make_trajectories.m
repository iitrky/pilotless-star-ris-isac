% Vehicle and target trajectories for the STAR-RIS junction scenario (Automated Driving Toolbox).
% Writes traj.csv with columns: t,id,x,y,z,vx,vy,kind  (consumed by scripts/sionna_export.py)
% Reflection-zone road: y = +10/+14 m; transmission-zone road: y = -10/-14 m (surface at y = 0).
Ts = 0.5e-3; T_end = 0.1;              % 200 slots per episode
rng(1);
scen = drivingScenario('SampleTime', Ts, 'StopTime', T_end);
road(scen, [-80 12 0; 80 12 0], 'Lanes', lanespec(2));
road(scen, [-80 -12 0; 80 -12 0], 'Lanes', lanespec(2));
lanesY = [10 14 -10 -14];
nVeh = 6; speeds = 30 + 90*rand(1, nVeh);          % km/h
for k = 1:nVeh
    v = vehicle(scen, 'ClassID', 1);
    y = lanesY(mod(k-1, 4) + 1); x0 = -60 + 120*rand;
    sgn = sign(randn); spd = speeds(k)/3.6;
    trajectory(v, [x0 y 0; x0 + sgn*spd*(T_end+0.01) y 0], spd);
end
tg = [actor(scen, 'ClassID', 4), actor(scen, 'ClassID', 4)];   % pedestrians/cyclists as targets
tgY = [20 -20];
for l = 1:2
    s = 1 + 4*rand; x0 = -30 + 60*rand;
    trajectory(tg(l), [x0 tgY(l) 0; x0 + s*(T_end+0.01) tgY(l) 0], s);
end
fid = fopen('traj.csv', 'w'); fprintf(fid, 't,id,x,y,z,vx,vy,kind\n');
while advance(scen)
    t = scen.SimulationTime;
    for a = scen.Actors
        if a.ClassID == 1, kind = 'vehicle'; z = 1.5; else, kind = 'target'; z = 1.0; end
        fprintf(fid, '%.6f,%d,%.4f,%.4f,%.2f,%.4f,%.4f,%s\n', t, a.ActorID, a.Position(1), a.Position(2), z, ...
            a.Velocity(1), a.Velocity(2), kind);
    end
end
fclose(fid);
