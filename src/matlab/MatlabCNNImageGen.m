%MATLAB Code for image generation to train the CNN model
% Specify the folder containing your .txt files
folderPath = 'C:\Users\namuz\OneDrive\Desktop\Cd2+ [Final]\0.05mMCd\Good\Txt'; % Adjust this to your folder path
% List all .txt files in the folder
txtFiles = dir(fullfile(folderPath, '*.txt'));
% Initialize an array to store unique identifiers and Y coordinates of the first sharp point
sharpPointsData = {};
% Desired image size (32x32)
desiredSize = [128, 128];
% Specify the factor by which to adjust the curve thickness
thicknessFactor = 0.3; % Adjust this value as needed
% Loop through each file
for fileIdx = 1:length(txtFiles)
    % Full path to the current .txt file
    filename = fullfile(txtFiles(fileIdx).folder, txtFiles(fileIdx).name);
    % Extract the unique identifier
    pattern = 'P\d+C\d+V\d+'; % Regular expression for the pattern P#C##V##
    [token, match] = regexp(filename, pattern, 'tokens', 'match');
    if isempty(match) % Skip if pattern not found
        disp(['Pattern not found in ', filename]);
        continue;
    end
    uniqueId = match{1}; % Assuming there's at least one match
    % Specify the delimiter and read the CV data
    delimiter = ','; % Change based on your file's formatting
    opts = detectImportOptions(filename, 'Delimiter', delimiter);
    data = readmatrix(filename, opts);
    % Check if the data has at least two columns
    if size(data, 2) < 2
        sharpPointsData{end+1, 1} = uniqueId; % Store unique identifier
        sharpPointsData{end, 2} = NaN; % Store NaN for Y coordinate
        continue;
    end
    % Adjust the current by multiplying with 10^12
    data(:,2) = data(:,2) * 10^12;
    % Assign columns to X and Y
    X = data(:,1);
    Y = data(:,2);
    % Initialize variables to store the first sharp point
    first_sharp_X = NaN;
    first_sharp_Y = NaN;
    % Iterate through the data to find the first sharp point
    for i = 2:length(X)
        if abs(X(i)) < abs(X(i-1))
            first_sharp_X = X(i);
            first_sharp_Y = Y(i);
            break; % Exit the loop after finding the first sharp point
        end
    end
    % Store the unique identifier and corresponding Y coordinate of the first sharp point
    sharpPointsData{end+1, 1} = uniqueId; % Store unique identifier
    sharpPointsData{end, 2} = first_sharp_Y; % Store Y coordinate
    % Plot with original X-axis order
    f1 = figure('Visible', 'off'); % Create invisible figure
    set(f1, 'Position', [0, 0, desiredSize(1), desiredSize(2)]); % Set figure size to desired size
    plot(X, Y, 'k', 'LineWidth', thicknessFactor); % Plot in black with adjusted thickness
    xlim([-0.75, -0.2]);
    ylim([-20, 60]);
    set(gca, 'PlotBoxAspectRatio', [1, 1, 1]); % Set aspect ratio to 1:1
    set(gca, 'Visible', 'off'); % Make the axis invisible    
    % Save the image
    originalImage = frame2im(getframe(gcf)); % Capture the figure as an image
    imwrite(originalImage, fullfile(folderPath, [uniqueId, '_original.png']));
    close(f1); % Close the figure to free up memory
    % Plot with reversed X-axis order
    f2 = figure('Visible', 'off'); % Create invisible figure
    set(f2, 'Position', [0, 0, desiredSize(1), desiredSize(2)]); % Set figure size to desired size
    plot(X, Y, 'k', 'LineWidth', thicknessFactor); % Plot in black with adjusted thickness
    set(gca, 'XDir','reverse'); % Reverse the X-axis direction
    xlim([-0.75, -0.2]); % Adjust the limits to reverse the direction
    ylim([-20, 60]);
    set(gca, 'PlotBoxAspectRatio', [1, 1, 1]); % Set aspect ratio to 1:1
    set(gca, 'Visible', 'off'); % Make the axis invisible    
    % Save the image
    reversedImage = frame2im(getframe(gcf)); % Capture the figure as an image
    imwrite(reversedImage, fullfile(folderPath, [uniqueId, '_reversed.png']));
    close(f2); % Close the figure to free up memory
end
% Convert the array to a table for better readability
sharpPointsTable = cell2table(sharpPointsData, 'VariableNames', {'UniqueIdentifier', 'Current'});
% Display the table
disp(sharpPointsTable);
