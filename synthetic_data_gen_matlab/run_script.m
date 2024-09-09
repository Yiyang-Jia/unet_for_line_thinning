% Specify the name of your existing .m file (without the .m extension)
existing_script = 'streda_line_thicken';
% Specify the name of your existing .m file (without the .m extension)


% Create directories for thick and thin line images
thick_dir = 'thick_lines_synthetic';
thin_dir = 'thin_lines_synthetic';
if ~exist(thick_dir, 'dir')
    mkdir(thick_dir);
end
if ~exist(thin_dir, 'dir')
    mkdir(thin_dir);
end


% Set figure visibility to 'off' before running the script
set(0, 'DefaultFigureVisible', 'off');
for u = 1: 20


    % Run the existing script
    run(existing_script);
    
    % Get the handles of the current figures
    figs = findall(0, 'Type', 'figure');
    
    % Ensure we have exactly two figures
    if length(figs) ~= 2
        error('Expected 2 figures, but got %d figures in run %d', length(figs), u);
    end
    
    % Generate filename
    filename = sprintf('run_%03d.png', u);
    
    % Save the first figure (assumed to be the thin lines) to the thin_lines directory
    saveas(figs(1), fullfile(thick_dir, filename));
    
    % Save the second figure (assumed to be the thick lines) to the thick_lines directory
    saveas(figs(2), fullfile(thin_dir, filename));
    
    % Close all figures
    close all;
    
    % Optional: Display progress
    fprintf('Completed run %d\n', u);
end
% Set figure visibility back to 'on' after running the script
set(0, 'DefaultFigureVisible', 'on');
fprintf('All runs completed. Images saved in %s and %s\n', thick_dir, thin_dir);