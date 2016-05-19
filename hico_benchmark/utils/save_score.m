function [] = save_score(file_name, res)

save(file_name, '-struct', 'res');

end