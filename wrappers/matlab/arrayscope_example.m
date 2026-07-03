%ARRAYSCOPE_EXAMPLE  ArrayScope from MATLAB — examples.
%
% Setup: add this folder to the MATLAB path and install the viewer:
%   addpath('/path/to/ArrayScope/wrappers/matlab')
%   % in a terminal: pip install arrayscope

% A synthetic complex 4-D "MRI-like" dataset: x, y, slice, coil.
[xg, yg] = ndgrid(linspace(-1, 1, 192), linspace(-1, 1, 160));
vol = zeros(192, 160, 12, 4, 'like', single(1i));
for s = 1:12
    for c = 1:4
        vol(:, :, s, c) = single(exp(-4 * (xg.^2 + yg.^2)) ...
            .* exp(1i * (6 * pi * xg * s / 8 + c)));
    end
end

% Non-blocking (default): MATLAB stays usable, the viewer opens detached.
arrayscope(vol, 'name', 'demo_kspace');

% Blocking, e.g. at the end of a batch script:
% arrayscope(abs(vol), 'name', 'demo_magnitude', 'block', true);

% Large-array tips:
% - Real arrays are written with a single fwrite (no compression) and
%   memory-mapped by the viewer; complex arrays cost one interleave pass.
% - On Linux: setenv('ARRAYSCOPE_HANDOFF_DIR', '/dev/shm') keeps the handoff
%   in RAM.
