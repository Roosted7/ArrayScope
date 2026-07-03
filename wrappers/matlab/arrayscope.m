function outpath = arrayscope(A, varargin)
%ARRAYSCOPE  View a numeric array in the ArrayScope viewer.
%
%   arrayscope(A)                        non-blocking; name from variable name
%   arrayscope(A, 'name', 'kspace')      custom window/handoff name
%   arrayscope(A, 'block', true)         wait until the viewer window closes
%   arrayscope(A, 'mmap', false)         eager read instead of memory-mapping
%   arrayscope(A, 'keep', true)          keep the handoff file after loading
%   arrayscope(A, 'dir', 'D:\fastssd')   custom handoff directory
%   arrayscope(A, 'exe', 'C:\...\arrayscope.exe')  explicit viewer executable
%   p = arrayscope(A)                    returns the handoff file path
%
%   The array is handed to a separate viewer process through a raw,
%   uncompressed NumPy .npy file:
%
%   - Real arrays are written with ONE fwrite of the underlying buffer — no
%     per-element work and, unlike save('-v7'), no zlib compression pass.
%     MATLAB's column-major layout is declared as fortran_order, so neither
%     side transposes or reorders anything.
%   - The viewer is launched with --mmap, so it memory-maps the file
%     (copy-on-write) rather than reading it eagerly; pages come straight
%     from the OS page cache the write just filled.
%   - --consume asks the viewer to delete the file once loaded (best effort
%     on Windows, where mapped files cannot be deleted; stale files are
%     cleaned up on the next arrayscope() call).
%   - Complex arrays need one interleave pass (MATLAB cannot fwrite complex
%     buffers directly); everything else is a single sequential write.
%   - On Linux, set ARRAYSCOPE_HANDOFF_DIR=/dev/shm to keep the handoff in
%     shared memory.
%
%   Requires the ArrayScope viewer on this machine:  pip install arrayscope
%   Resolution order for the executable: 'exe' argument, ARRAYSCOPE_EXE
%   environment variable, 'arrayscope' on PATH.
%
%   Works on Windows, macOS, and Linux. No MATLAB-Python (pyenv) setup is
%   needed: the viewer runs in its own process.

    % --- Parse arguments ---
    defname = inputname(1);
    if isempty(defname)
        defname = 'array';
    end
    p = inputParser;
    addParameter(p, 'name', defname, @(x) ischar(x) || isstring(x));
    addParameter(p, 'block', false, @islogical);
    addParameter(p, 'mmap', true, @islogical);
    addParameter(p, 'keep', false, @islogical);
    addParameter(p, 'dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'exe', '', @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});
    opts = p.Results;
    name = char(opts.name);

    % --- Input validation ---
    validateattributes(A, {'numeric', 'logical'}, {'nonempty'}, 'arrayscope', 'A');

    % --- Handoff directory ---
    d = char(opts.dir);
    if isempty(d)
        d = getenv('ARRAYSCOPE_HANDOFF_DIR');
    end
    if isempty(d)
        d = fullfile(tempdir, 'arrayscope-handoff');
    end
    if ~exist(d, 'dir')
        mkdir(d);
    end
    clean_stale_handoffs(d);

    % --- Write the raw .npy handoff file ---
    safe = regexprep(name, '[^A-Za-z0-9_.-]', '_');
    if isempty(safe)
        safe = 'array';
    end
    fname = sprintf('%s-%d-%04d.npy', safe, round(posixtime(datetime('now')) * 1000), ...
                    randi(9999));
    outpath = fullfile(d, fname);
    write_npy(outpath, A);

    % --- Resolve the viewer executable ---
    exe = char(opts.exe);
    if isempty(exe)
        exe = getenv('ARRAYSCOPE_EXE');
    end
    if isempty(exe)
        exe = 'arrayscope';
    end

    % --- Build and launch the viewer command ---
    flags = sprintf(' --title "%s"', strrep(name, '"', ''));
    if opts.mmap
        flags = [flags ' --mmap'];
    end
    if ~opts.keep
        flags = [flags ' --consume'];
    end
    cmd = sprintf('"%s"%s "%s"', exe, flags, outpath);

    if opts.block
        status = system(cmd);
        if status ~= 0
            error('arrayscope:launchFailed', ...
                  ['ArrayScope viewer exited with status %d.' newline ...
                   'Is it installed? Try:  pip install arrayscope' newline ...
                   'Or set the ARRAYSCOPE_EXE environment variable.'], status);
        end
    else
        if ispc
            status = system(['start "" /b ' cmd ' >NUL 2>&1']);
        else
            status = system([cmd ' >/dev/null 2>&1 &']);
        end
        if status ~= 0
            error('arrayscope:launchFailed', ...
                  ['Could not launch the ArrayScope viewer.' newline ...
                   'Is it installed? Try:  pip install arrayscope' newline ...
                   'Or set the ARRAYSCOPE_EXE environment variable.']);
        end
    end

    if nargout == 0
        clear outpath
    end
end


function write_npy(path, A)
%WRITE_NPY Write a dense array as an uncompressed NumPy .npy (format 1.0).
%   Column-major data is written verbatim and declared fortran_order: True.
%   Real data: one fwrite of the buffer. Complex data: one interleave pass.

    [descr, precision] = npy_descr(A);

    % MATLAB pads everything to >= 2 dims; drop trailing singletons so a
    % column vector arrives in Python as shape (n,) rather than (n, 1).
    sz = size(A);
    while numel(sz) > 1 && sz(end) == 1
        sz(end) = [];
    end
    if numel(sz) == 1
        shapestr = sprintf('(%d,)', sz(1));
    else
        shapestr = ['(' sprintf('%d, ', sz(1:end-1)) sprintf('%d)', sz(end))];
    end

    dict = sprintf('{''descr'': ''%s'', ''fortran_order'': True, ''shape'': %s, }', ...
                   descr, shapestr);
    % magic(6) + version(2) + header-length field(2) + header, space-padded so
    % the data section starts on a 64-byte boundary; header ends with \n.
    unpadded = 6 + 2 + 2 + numel(dict) + 1;
    pad = mod(64 - mod(unpadded, 64), 64);
    header = [dict repmat(' ', 1, pad) newline];

    fid = fopen(path, 'w', 'ieee-le');  % force little-endian on every platform
    if fid < 0
        error('arrayscope:io', 'Could not open handoff file: %s', path);
    end
    cleaner = onCleanup(@() fclose(fid));

    fwrite(fid, uint8([147 'NUMPY']), 'uint8');   % \x93NUMPY
    fwrite(fid, uint8([1 0]), 'uint8');           % format 1.0
    fwrite(fid, uint16(numel(header)), 'uint16');
    fwrite(fid, header, 'char');

    if ~isreal(A)
        % .npy complex layout is interleaved re/im pairs; MATLAB cannot
        % fwrite complex buffers, so interleave once (the only extra pass).
        ri = zeros(2, numel(A), precision);
        ri(1, :) = real(A(:));
        ri(2, :) = imag(A(:));
        fwrite(fid, ri, precision);
    elseif islogical(A)
        fwrite(fid, uint8(A), 'uint8');           % numpy bool is 1 byte
    else
        fwrite(fid, A, precision);                % raw buffer, one pass
    end
end


function [descr, precision] = npy_descr(A)
    cls = class(A);
    cplx = ~isreal(A);
    switch cls
        case 'double'
            precision = 'double';
            if cplx, descr = '<c16'; else, descr = '<f8'; end
        case 'single'
            precision = 'single';
            if cplx, descr = '<c8'; else, descr = '<f4'; end
        case 'logical'
            precision = 'uint8';  descr = '|b1';
        case 'int8'
            precision = cls; descr = '|i1';
        case 'uint8'
            precision = cls; descr = '|u1';
        case 'int16'
            precision = cls; descr = '<i2';
        case 'uint16'
            precision = cls; descr = '<u2';
        case 'int32'
            precision = cls; descr = '<i4';
        case 'uint32'
            precision = cls; descr = '<u4';
        case 'int64'
            precision = cls; descr = '<i8';
        case 'uint64'
            precision = cls; descr = '<u8';
        otherwise
            error('arrayscope:unsupportedType', ...
                  'Unsupported array class: %s', cls);
    end
    if cplx && ~any(strcmp(cls, {'double', 'single'}))
        error('arrayscope:unsupportedType', ...
              'Complex integer arrays are not supported.');
    end
end


function clean_stale_handoffs(d)
%CLEAN_STALE_HANDOFFS Best-effort removal of handoff files older than 24h
%   (files the viewer could not --consume, e.g. on Windows).
    try
        files = dir(fullfile(d, '*.npy'));
        for k = 1:numel(files)
            if (now - files(k).datenum) * 86400 > 86400  %#ok<TNOW1>
                try
                    delete(fullfile(d, files(k).name));
                catch
                end
            end
        end
    catch
    end
end
