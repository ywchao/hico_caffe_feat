function w_cand = get_w_cand( num_w )
% generate hard-coded weight candidates

switch num_w
    case 1
        w_cand = 1;
    case 2
        w1_cand = 0:0.01:1;
        w_cand = zeros(numel(w1_cand),1);
        for i = 1:numel(w1_cand)
            w_cand(i,1) = w1_cand(i);
            w_cand(i,2) = 1 - w1_cand(i);
        end
    case 3
        w1_cand = 0:0.01:1;
        w2_cand = 0:0.01:1;
        w_cand = [];
        for i = 1:numel(w1_cand)
            for j = 1:numel(w2_cand)
                if w1_cand(i) + w2_cand(j) > 1
                    break
                end
                w_cand = [w_cand; ...
                    w1_cand(i) ...
                    w2_cand(j) ...
                    1-w1_cand(i)-w2_cand(j)];
            end
        end
    case 4
        w1_cand = 0:0.025:1;
        w2_cand = 0:0.025:1;
        w3_cand = 0:0.025:1;
        w_cand = [];
        for i = 1:numel(w1_cand)
            for j = 1:numel(w2_cand)
                for k = 1:numel(w3_cand)
                    if w1_cand(i) + w2_cand(j) + w3_cand(k) > 1
                        break
                    end
                    w_cand = [w_cand; ...
                        w1_cand(i) ...
                        w2_cand(j) ...
                        w3_cand(k) ...
                        1-w1_cand(i)-w2_cand(j)-w3_cand(k)];
                end
            end
        end
    case 5
        w1_cand = 0:0.05:1;
        w2_cand = 0:0.05:1;
        w3_cand = 0:0.05:1;
        w4_cand = 0:0.05:1;
        w_cand = [];
        for i = 1:numel(w1_cand)
            for j = 1:numel(w2_cand)
                for k = 1:numel(w3_cand)
                    for l = 1:numel(w4_cand)
                        if w1_cand(i) + w2_cand(j) + w3_cand(k) + w4_cand(l) > 1
                            break
                        end
                        w_cand = [w_cand; ...
                            w1_cand(i) ...
                            w2_cand(j) ...
                            w3_cand(k) ...
                            w4_cand(l) ...
                            1-w1_cand(i)-w2_cand(j)-w3_cand(k)-w4_cand(l)];
                    end
                end
            end
        end
    case 6
        w1_cand = 0:0.075:1;
        w2_cand = 0:0.075:1;
        w3_cand = 0:0.075:1;
        w4_cand = 0:0.075:1;
        w5_cand = 0:0.075:1;
        w_cand = [];
        for i = 1:numel(w1_cand)
            for j = 1:numel(w2_cand)
                for k = 1:numel(w3_cand)
                    for l = 1:numel(w4_cand)
                        for m = 1:numel(w5_cand)
                            if w1_cand(i) + w2_cand(j) + w3_cand(k) + w4_cand(l) + w5_cand(m) > 1
                                break
                            end
                            w_cand = [w_cand; ...
                                w1_cand(i) ....
                                w2_cand(j) ...
                                w3_cand(k) ...
                                w4_cand(l) ...
                                w5_cand(m) ...
                                1-w1_cand(i)-w2_cand(j)-w3_cand(k)-w4_cand(l)-w5_cand(m)];
                        end
                    end
                end
            end
        end
    case 7
        w1_cand = 0:0.08:1;
        w2_cand = 0:0.08:1;
        w3_cand = 0:0.08:1;
        w4_cand = 0:0.08:1;
        w5_cand = 0:0.08:1;
        w6_cand = 0:0.08:1;
        w_cand = [];
        for i = 1:numel(w1_cand)
            for j = 1:numel(w2_cand)
                for k = 1:numel(w3_cand)
                    for l = 1:numel(w4_cand)
                        for m = 1:numel(w5_cand)
                            for n = 1:numel(w6_cand)
                                if w1_cand(i) + w2_cand(j) + w3_cand(k) + w4_cand(l) + w5_cand(m) + w6_cand(n) > 1
                                    break
                                end
                                w_cand = [w_cand; ...
                                    w1_cand(i) ...
                                    w2_cand(j) ...
                                    w3_cand(k) ...
                                    w4_cand(l) ...
                                    w5_cand(m) ...
                                    w6_cand(n) ...
                                    1-w1_cand(i)-w2_cand(j)-w3_cand(k)-w4_cand(l)-w5_cand(m)-w6_cand(n)];
                            end
                        end
                    end
                end
            end
        end
    % case 8
    %     w1_cand = 0:0.1:1;
    %     w2_cand = 0:0.1:1;
    %     w3_cand = 0:0.1:1;
    %     w4_cand = 0:0.1:1;
    %     w5_cand = 0:0.1:1;
    %     w6_cand = 0:0.1:1;
    %     w7_cand = 0:0.1:1;
    %     w_cand = [];
    %     for i = 1:numel(w1_cand)
    %         for j = 1:numel(w2_cand)
    %             for k = 1:numel(w3_cand)
    %                 for l = 1:numel(w4_cand)
    %                     for m = 1:numel(w5_cand)
    %                         for n = 1:numel(w6_cand)
    %                             for o = 1:numel(w7_cand)
    %                                 if w1_cand(i) + w2_cand(j) + w3_cand(k) + w4_cand(l) + w5_cand(m) + w6_cand(n) + w7_cand(o)> 1
    %                                     break
    %                                 end
    %                                 w_cand = [w_cand; ...
    %                                     w1_cand(i) ....
    %                                     w2_cand(j) ...
    %                                     w3_cand(k) ...
    %                                     w4_cand(l) ...
    %                                     w5_cand(m) ...
    %                                     w6_cand(n) ...
    %                                     w7_cand(o) ...
    %                                     1-w1_cand(i)-w2_cand(j)-w3_cand(k)-w4_cand(l)-w5_cand(m)-w6_cand(n)-w7_cand(o)];
    %                             end
    %                         end
    %                     end
    %                 end
    %             end
    %         end
    %     end
end

w_cand = max(w_cand,0);
    
end

