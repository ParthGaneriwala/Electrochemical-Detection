%MATLAB Code for data generation and preprocessing to train the ANN model
% Specify the folder containing your .txt files
folderPath = 'C:\Users\namuz\OneDrive\Desktop\retxt'; % Adjust this to your folder path
% List all .txt files in the folder
txtFiles = dir(fullfile(folderPath, '*.txt'));
% Initialize a cell array to store the table data
tableData = cell(length(txtFiles), 7);
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
    uniqueId = match{1}; 
    % Specify the delimiter and read the CV data
    delimiter = ','; % Change based on your file's formatting
    opts = detectImportOptions(filename, 'Delimiter', delimiter);
    data = readmatrix(filename, opts);
    % Check if the data has at least two columns
    if size(data, 2) < 2
        disp(['Data in ', filename, ' does not have two columns. Skipping.']);
        continue;
    end
    % Adjust the current by multiplying with 10^12
    data(:,2) = data(:,2) * 10^12;
    % Assign columns to X and Y
    X = data(:,1);
    Y = data(:,2);
    % Identify the direction change point in X
    dX = diff(X);
    changePointIdx = find(dX(1:end-1) .* dX(2:end) < 0, 1, 'first') + 1;
    if isempty(changePointIdx)
        disp(['No change in direction detected in ', filename, '. Skipping.']);
        continue;
    end    
    % Extract the sharp point values
    sharpPointX = X(changePointIdx);
    sharpPointY = Y(changePointIdx);    
    % Separate into forward and backward scans
    X_forward = X(1:changePointIdx);
    Y_forward = Y(1:changePointIdx);
    X_backward = X(changePointIdx:end);
    Y_backward = Y(changePointIdx:end);
    % Now plot using X_forward and Y_forward
    %figure;
    %plot(X_backward, Y_backward, 'g-', 'LineWidth', 2); % Plot backward scan in green
    %hold on;
    %plot(X_forward, Y_forward, 'b-', 'LineWidth', 2); % Plot forward scan in blue
    %xlabel('Potential (V)');
    %ylabel('Current (pA)');
    %title(['Current vs. Potential for ', uniqueId]);
    %grid on;
    %xlabel('X-Axis');
    %ylabel('Y-Axis');
    %legend('Data', 'First Sharp Point');
    %xlim([-0.65, -0.2]);
    %ylim([-20, 70]);
    % Calculate the derivative of Y in the forward scan
    dy_forward = diff(Y_forward);
    
    % Set a threshold for derivative values close to zero for plateau detection
    derivative_threshold = 8e-2;    
    % Set a threshold for the x-axis distance between plateau points and sharp point
    X_threshold = 0.1;    
    % Find the indices where the derivative in the forward scan is close to zero
    % and the x-values are above the x-threshold
    plateau_indices_forward = find(abs(dy_forward) < derivative_threshold & abs(X_forward(1:end-1) - sharpPointX) >= X_threshold);   
    % Extract the coordinates of the last plateau value in the forward scan
    if ~isempty(plateau_indices_forward)
        lastPlateauX = X_forward(plateau_indices_forward(end));
        lastPlateauY = Y_forward(plateau_indices_forward(end));
    else
        lastPlateauX = NaN;
        lastPlateauY = NaN;
    end    
    % Calculate the derivative of Y in the reverse scan
    dy_backward = diff(Y_backward);
    % Find the indices where the derivative in the reverse scan is close to zero
    % and the x-values are above the x-threshold
    plateau_indices_backward = find(abs(dy_backward) < derivative_threshold & abs(X_backward(2:end) - sharpPointX) >= X_threshold);
    % Extract the coordinates of the last plateau value in the reverse scan
    if ~isempty(plateau_indices_backward)
        lastPlateauX_backward = X_backward(plateau_indices_backward(1) + 1);
        lastPlateauY_backward = Y_backward(plateau_indices_backward(1) + 1);
    else
        lastPlateauX_backward = NaN;
        lastPlateauY_backward = NaN;
    end    
    % Store the data in the table
    tableData{fileIdx, 1} = uniqueId;
    tableData{fileIdx, 2} = sharpPointX;
    tableData{fileIdx, 3} = sharpPointY;
    tableData{fileIdx, 4} = lastPlateauX;
    tableData{fileIdx, 5} = lastPlateauY;
    tableData{fileIdx, 6} = lastPlateauX_backward;
    tableData{fileIdx, 7} = lastPlateauY_backward;    
    % Save the sharp point and last plateau point values in a new text file
    newFileName = ['shrBK', uniqueId, '.txt'];
    newFilePath = fullfile(folderPath, newFileName);
    fileID = fopen(newFilePath, 'w');
    fprintf(fileID, '%.6f,%.6f\n', sharpPointX, sharpPointY);
    fprintf(fileID, '%.6f,%.6f\n', lastPlateauX, lastPlateauY);
    fprintf(fileID, '%.6f,%.6f\n', lastPlateauX_backward, lastPlateauY_backward);
    fclose(fileID);    
    % Identify the plateau region in terms of X values in the forward scan
    if ~isempty(plateau_indices_forward)
        plateau_X_forward = X_forward(plateau_indices_forward);
        % Visualize the detected plateau region in the forward scan
        %plot(plateau_X_forward, Y_forward(plateau_indices_forward), 'ro', 'MarkerSize', 8); % Mark plateau points
    else
        disp(['No plateau found in forward scan of ', filename]);
    end    
    % Identify the plateau region in terms of X values in the reverse scan
    if ~isempty(plateau_indices_backward)
        plateau_X_backward = X_backward(plateau_indices_backward + 1);
        % Visualize the detected plateau region in the reverse scan
        %plot(plateau_X_backward, Y_backward(plateau_indices_backward + 1), 'mo', 'MarkerSize', 8); % Mark plateau points
        %legend('Backward Scan', 'Forward Scan', 'Plateau Region (Forward)', 'Sharp Point', 'Plateau Region (Reverse)');
    else
        disp(['No plateau found in reverse scan of ', filename]);
        %legend('Backward Scan', 'Forward Scan', 'Plateau Region (Forward)', 'Sharp Point');
    end    
    % Plot the sharp point
    %plot(sharpPointX, sharpPointY, 'ko', 'MarkerSize', 10, 'LineWidth', 2);
    %hold off;
end
% Create a table with the collected data
varNames = {'File Name', 'Sharp Point X', 'Sharp Point Y', 'Last Plateau X (Forward)', 'Last Plateau Y (Forward)', 'Last Plateau X (Reverse)', 'Last Plateau Y (Reverse)'};
resultTable = cell2table(tableData, 'VariableNames', varNames);
% Display the table
disp(resultTable);
