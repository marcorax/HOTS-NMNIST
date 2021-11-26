%% Load train and test sets
str_train="Train/Train/";
str_test="Test/Test/";
train_set=cell(10,1);
test_set=cell(10,1);
res_x = 34;
res_y = 34;

for i=1:10
    train_set_single_class={};
    test_set_single_class={};
    folder_train = str_train+string(i-1)+'/';
    folder_test = str_test+string(i-1)+'/';
    listdir_train=dir(folder_train);
    listdir_test=dir(folder_test);
    for j=1:length(listdir_train)
        if listdir_train(j).isdir==0
            train_set_single_class=cat(2,stabilize(Read_Ndataset(folder_train+listdir_train(j).name)),train_set_single_class);    
        end
    end
    for j=1:length(listdir_test)
        if listdir_test(j).isdir==0
            test_set_single_class=cat(2,stabilize(Read_Ndataset(folder_test+listdir_test(j).name)),test_set_single_class);   
        end
    end
    % Need to cast to save memory
    n_recordings = length(train_set_single_class);
    parfor recording=1:n_recordings
        train_set_single_class(recording).x = cast(train_set_single_class(recording).x,'uint8');
        train_set_single_class(recording).y = cast(train_set_single_class(recording).y,'uint8');
        train_set_single_class(recording).p = cast(train_set_single_class(recording).p,'uint8');
        train_set_single_class(recording).ts = cast(train_set_single_class(recording).ts,'single');
    end
    n_recordings = length(test_set_single_class);
    parfor recording=1:n_recordings
        test_set_single_class(recording).x = cast(test_set_single_class(recording).x,'uint8');
        test_set_single_class(recording).y = cast(test_set_single_class(recording).y,'uint8');
        test_set_single_class(recording).p = cast(test_set_single_class(recording).p,'uint8');
        test_set_single_class(recording).ts = cast(test_set_single_class(recording).ts,'single');
    end
    train_set(i)={train_set_single_class};
    test_set(i)={test_set_single_class};
end

save('train_set.mat','train_set')
save('test_set.mat','test_set')