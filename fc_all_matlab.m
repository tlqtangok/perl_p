*********************
./tma_detect/src/tma.m
*********************
main();

if 0
x_set = 0;
for i = 1:1000
    x = sqrt(0.5*i*i + 1/i); 
    x_set(i) = x;  
end

x_set

d = 50
for i = 1:4:100
rnd = randi(1000) 
x_set(rnd) = x_set(rnd) + mod(rnd,d) - d/2;
end
y_s=movmean(x_set,5);
y_ss=movmean(y_s,5); 
y_ss = movmean(y_ss, 33); 
plot(y_s); hold on ; plot(x_set); hold on; plot(y_ss);
plot(1:1000, x_set, ".");
axis equal;
end 
function m = main()
% GLOBAL VAR %
%%%%%%%%%%%%%%
loc_uav_r0=[0,0,0];
R = 100;
obj_theta = 15;


v_uav = [1,0,0];
v_obj = [0.2 -0.8 0];


speed_val_uav = 0;
speed_val_obj = 0.1;


loc_uav_set=[];
loc_obj_set=[];
tan_t_set=[];
%%%%%%%%%%%%%%



loc_obj_r0=[R.*sind(obj_theta),R.*cosd(obj_theta),0];
dt = 1;
v_uav = v_uav ./ sqrt(dot(v_uav, v_uav)) .* speed_val_uav;
v_obj = v_obj ./ sqrt(dot(v_obj, v_obj)) .* speed_val_obj;


loc_uav = loc_uav_r0;
loc_obj = loc_obj_r0;

idx_cnt_uav = 1;
idx_cnt_obj = 1;
idx_cnt_tan_t = 1;

pos_edge = [90*0,90*1,90*2, 90*3]; 
idx_pos_edge = 0; 
minus_theta = 1;
theta_set = [];
for step_cnt = 1:20000
    
    v_d = loc_obj - loc_uav;
    dx = v_d(1,1);
    dy = v_d(1,2);
    tan_t = dx./dy;
    theta = atand(tan_t);
    theta_set(step_cnt) = theta;
    
    
    if step_cnt == 1
        minus_theta = 1;
    else
        minus_theta = theta * theta_set(step_cnt-1);
    end
    
    if minus_theta < 0
        idx_pos_edge = idx_pos_edge + 1;
    end
    

    
    %     tan_t = log(tan_t);
    
    tan_t_set(idx_cnt_tan_t,:) = [theta + 180 * idx_pos_edge ,idx_cnt_tan_t,0];
    idx_cnt_tan_t= idx_cnt_tan_t+1;
    
    loc_uav_set(idx_cnt_uav,:) = loc_uav;
    idx_cnt_uav = idx_cnt_uav + 1;
    loc_obj_set(idx_cnt_obj,:) = loc_obj;
    idx_cnt_obj = idx_cnt_obj + 1;
    
    loc_uav = v_uav.*dt + loc_uav;
    loc_obj = v_obj.*dt + loc_obj;
    

    
end




%%% plot area
%%%%%%%%%%%%%
% plot_p3d_set(loc_uav_set , 'O');
% plot_p3d_set(loc_obj_set , '*');
plot_p3d_set(tan_t_set, ".");
view([0,0,1]);


%%%%%%%%%%%%%

end



function p= plot_p3d_set(p3d_set, style)
grid;
axis equal;
plot3(p3d_set(:,1), p3d_set(:,2), p3d_set(:,3),style);
hold on;
end


function angle = uniform_degree(x,y)
if x>0 && y>0
else if x>0 && y< 0 
    else if x<0 and y<0
        else
        end
        
end



*********************
./uav_deviation_correct/src/call_uav_control.m
*********************
cur_loc = [1,0,0];

road_mark = [[100,0,0]; [33,7,0]; [66,-66,0]; [99,44,0];[55,-11,0];[-11,54,0];[88,-77,0];[-1,-1,0]];
road_mark_verify = [[ 78.375,-24.75,0 ];[ 16.8,-16.2,0 ];[ 20.8,0,0 ]];

cur_loc_set = [];
sz_dis_last_few = 7;
dis_near_range = 3;
dis_verify_range = 6;
speed_val = 1;
factor_ocean_d_uav = 0.000001;

max_n_dis_deviation = 1;

if 1
    
p3d_set = [-61.0000   22.0000         0;  -50.5814   33.3816         0;  -55.0000   26.0000         0;  -59.0000   44.0000         0];
cur_loc = p3d_set(1,:);

a=p3d_set(2,:);
b=p3d_set(3,:);
c=p3d_set(4,:);

road_mark=[a;b;c]; 

%road_mark_verify = [[78.375  -24.75 0]; [16.8 -16.2 0];[20.8 0 0 ]];
road_mark_verify = [];
lambda = 0.7;
road_mark_verify(1,:) = a.*lambda + b.*(1-lambda);
lambda = 0.5;
road_mark_verify(2,:) = b.*lambda + c.*(1-lambda);

dis_near_range = 0.1;
dis_verify_range = 0.2;
max_n_dis_deviation = 0.3;
speed_val = 1/400;


end 


%road_mark_prev = cur_loc;
road_mark_next = road_mark(1,:);





dis_last_few = ones(1,sz_dis_last_few).*dis_near_range;


road_mark_prev = []
idx_node_visit = 1;
idx_cur_loc = 1;


    
plot_p3d_set(cur_loc, '-.');
text(cur_loc(1,1)+1,cur_loc(1,2),cur_loc(1,3), int2str(0));
hold on;

if sum(size(road_mark_verify)) ~= 0
for i=1:length(road_mark_verify(:,1))
plot_p3d(road_mark_verify(i,:), 'x');
end
end

for i=1:length(road_mark(:,1))
plot_p3d(road_mark(i,:), 'O');
text(road_mark(i,1)+1,road_mark(i,2),road_mark(i,3), int2str(i)); 
end

flag_finish_all_road_mark = 0;
flag_arrived_dst = 0;
flag_goto_set_cur_loc = 0;
flag_deviation = 0;
flag_need_re_cal_v_sum = 0;


cur_loc_r0 = cur_loc;


flag_arrived_to_verify_point = 0;

v_sum = [];

for step_cnt = 1:20000
    

    flag_need_re_cal_v_sum = 0;
    flag_goto_set_cur_loc = 0;
    
    cur_loc_set(idx_cur_loc,:) = cur_loc;
    cur_loc_r0 = cur_loc;
    
    idx_cur_loc = idx_cur_loc+1;
    
    road_mark_next = road_mark(idx_node_visit,:);
    
    if step_cnt == 1 || flag_deviation == 1 || flag_arrived_to_verify_point == 1 || flag_arrived_dst == 1
        flag_need_re_cal_v_sum = 1;
    end
    
    cur_loc_new = [0,0,0];
    [cur_loc_new, v_sum, dis_last_few, flag_arrived_dst, flag_deviation,flag_arrived_to_verify_point,road_mark_verify_center, ...
        road_mark_prev, road_mark_verify] ...
        ...
        = uav_control(step_cnt, v_sum, flag_need_re_cal_v_sum,  cur_loc, road_mark_prev, road_mark_next, speed_val, ...
        ...
        factor_ocean_d_uav,dis_last_few, dis_near_range, flag_deviation, max_n_dis_deviation, road_mark_verify, dis_verify_range);
    
    cur_loc = cur_loc_new;
    
   



    if flag_arrived_dst == 1
        idx_node_visit = idx_node_visit+1;
                
        if idx_node_visit > length(road_mark(:,1))
            "- end all node"        
            flag_finish_all_road_mark = 1;
            break;
        end    
        
        road_mark_next_new = road_mark(idx_node_visit,:);
        road_mark_prev_new = road_mark_next;
        [cur_loc, road_mark_prev, road_mark_next ] = arrived_and_switch_node(flag_arrived_dst, cur_loc, road_mark_prev, road_mark_next, road_mark_prev_new, road_mark_next_new);
        %flag_arrived_dst = 0;
        flag_goto_set_cur_loc = 1;
    end

    
    
    
    
    if flag_arrived_to_verify_point == 1
        
        [road_mark] = update_road_mark_when_meet_verify(cur_loc, road_mark_verify_center, road_mark, idx_node_visit);
        road_mark;
        %flag_arrived_to_verify_point = 0;
        flag_goto_set_cur_loc = 1;
    end
    

    if flag_goto_set_cur_loc == 1
        flag_goto_set_cur_loc = 0;
    else
     [cur_loc] = set_back_cur_loc(cur_loc, cur_loc_r0);
    end

end





plot_p3d_set(cur_loc_set, '-.');






view([0,0,1]);

axis equal;

function [cur_loc] = set_back_cur_loc(cur_loc, cur_loc_r0)
%cur_loc(1, 3) = cur_loc_r0(1,3);
end
function [road_mark] = update_road_mark_when_meet_verify(cur_loc, road_mark_verify_center, road_mark, idx_node_visit)
C = cal_verify_edge_point(cur_loc,road_mark_verify_center);
%    / D
%   /B 
% A/_.._\C
% <A = 45 <B = 90 C=45
%D = B.*2 - A
D = road_mark_verify_center.*2 - cur_loc;

r_front = [];
if idx_node_visit>=2
    r_front = road_mark(1:idx_node_visit-1,:);
end
r_end = road_mark(idx_node_visit:end, :);
road_mark_new = [r_front; C; D; r_end];

road_mark = road_mark_new;        
end

function [cur_loc, road_mark_prev, road_mark_next ] = arrived_and_switch_node(flag_arrived_dst, cur_loc, road_mark_prev, road_mark_next, road_mark_prev_new, road_mark_next_new)
assert(flag_arrived_dst == 1); 
road_mark_prev_old = road_mark_prev;
road_mark_next_old = road_mark_next;

cur_loc = road_mark_prev_new;

road_mark_prev = road_mark_prev_new; 
road_mark_next = road_mark_next_new;

end

function p= plot_p3d_set(p3d_set, style)

axis equal;
plot3(p3d_set(:,1), p3d_set(:,2), p3d_set(:,3),style);
hold on;
end
function p=plot_p3d(cur_loc, style)
plot3(cur_loc(1), cur_loc(2), cur_loc(3),style);
hold on;
grid;
end



function pointC = cal_verify_edge_point(A,B)
cur_loc = A; 
road_mark_verify_center = B;
x0=A(1);
y0=A(2);
z0=A(3);
x1=B(1);
y1=B(2);
z1=B(3);

z2=z1;
%syms x0 y0 z0 x1 y1 z1 x2 y2 z2;
%A=solve([z2==z1, (x1-x0)*(x2-x1)+(y2-y1)*(y1-y0)+(z2-z1)*(z1-z0) == 0, (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0) == (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1) ],[x2,y2,z2]); 
x2 = x1 - y0*((x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2 + z0^2 - 2*z0*z1 + z1^2)/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2))^(1/2) + y1*((x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2 + z0^2 - 2*z0*z1 + z1^2)/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2))^(1/2)

x2 =  -(y0*(y1 + x0*((x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2 + z0^2 - 2*z0*z1 + z1^2)/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2))^(1/2) - x1*((x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2 + z0^2 - 2*z0*z1 + z1^2)/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2))^(1/2)) - y1*(y1 + x0*((x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2 + z0^2 - 2*z0*z1 + z1^2)/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2))^(1/2) - x1*((x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2 + z0^2 - 2*z0*z1 + z1^2)/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2))^(1/2)) - x0*x1 - y0*y1 + x1^2 + y1^2)/(x0 - x1);
y2 = y1 + x0*((x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2 + z0^2 - 2*z0*z1 + z1^2)/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2))^(1/2) - x1*((x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2 + z0^2 - 2*z0*z1 + z1^2)/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2))^(1/2);

pointC = [x2,y2,z2];
end

*********************
./uav_deviation_correct/src/call_v_ship_v_ocean_mix.m
*********************
% function v_ship_v_ocean_mix_ = v_ship_v_ocean_mix(loc_ship , v_ship_t0, v_ocean_t0)
% 
% %v_ship_v_ocean_mix_;
% 
% for i = 1:100
%     dt =1 * i;
%     
%     v_all = v_ship_t0 + v_ocean_t0; % t0
%     
%     v_all_dt = v_all * dt; %
%     
%     v_ship_v_ocean_mix_ = loc_ship  + v_all_dt; %t1
% end
% 
% end


loc_ship = [1,2,0];
v_ship_t0 = [100,111,0];
v_ocean_t0 = [1,2.3,0];


ship_loc_list = v_ship_v_ocean_mix(loc_ship , v_ship_t0, v_ocean_t0);
ship_loc_list; 

plot3(ship_loc_list(:,1), ship_loc_list(:,2), ship_loc_list(:,3), '-.')

*********************
./uav_deviation_correct/src/corr_direct.m
*********************

start_i = 2;
end_i = 7;

MAX_DEVIATION_DISTANCE = 3

dis_loc_to_drv = [0,0,0]
idx_node_next = 0;


flag_drv_directly = 0;
flag_normal_deviation = 0;

%x = round(sort(rand(10,1))*77); y = round(rand(10,1)*10); z = round(rand(10,1)*4); csvwrite("p3d.csv", [x,y,z]); 
%t = round(sort(randi(300,200,1)));tu = unique(t); x=tu(1:100); y = round(rand(100,1)*10);z = round(rand(100,1)*4); csvwrite("p3d_100.csv", [x,y,z])
grid;
road_mark = csvread("../data/p3d_road_mark_10.csv");
%road_mark = csvread("d:\\jd\\t\\1.csv");
%road_mark = csvread("../data/p3d_road_mark_sim_4.csv");
assert(start_i<=length(road_mark) && start_i>=1);
assert(end_i<=length(road_mark) && end_i>=1);
assert(start_i<=end_i);


r_mark_x = road_mark(:,1);
r_mark_y = road_mark(:,2);
r_mark_z = road_mark(:,3);

idx_rnd = randperm(length(r_mark_x)-1);
x_drv = r_mark_x(idx_rnd(1)) + rand(1)* 20;
y_drv = r_mark_y(idx_rnd(1)) + rand(1)* 8;
z_drv = r_mark_z(idx_rnd(1)) + rand(1)* 4;
xyz_drv = [x_drv, y_drv, z_drv];

if 1
    
    xyz_drv = [24.97,2.39,5.21];
    %xyz_drv = [2030211.83,-5496285.54,2511415.82]
    %xyz_drv = [2200942.3108777641,-4995430.2130303187,3287751.8454419416];
    %xyz_drv = [2033171.6464151894 5494368.0815875614 2513204.3387343464 ]

    x_drv = xyz_drv(1);
    y_drv = xyz_drv(2);
    z_drv = xyz_drv(3);
    % "drv(48.11,8.25,2.91) => (node(8)~node(9)) at (47.48,3.89,2.37), ...-> node(9)"
end


xyz_drv

dis_list = ones(length(r_mark_x),1);
dis_list = dis_list.*(100);
%dis_list




dis_list_ = sqrt((r_mark_x-x_drv).^2 +  (r_mark_y-y_drv).^2 + (r_mark_z-z_drv).^2);

for i = start_i:end_i
    dis_list(i) = dis_list_(i);
end

dis_list


%dis_list = dis_list(start_i:end_i,:);


loc_dis_list = [ (1:length(dis_list))', dis_list];


loc_dis_list

loc_dis_list_R0 = loc_dis_list;

loc_dis_list_sort = sortrows(loc_dis_list, 2);
loc_dis_list_sort





near_node_idx_best = loc_dis_list_sort(1,1);
near_node_idx_subopt = loc_dis_list_sort(2,1);

near_idx_head = max(near_node_idx_best, near_node_idx_subopt);
near_idx_tail = min(near_node_idx_best, near_node_idx_subopt);

% if the two nodes is adjacent ? 
flag_need_divide_angle = 1;
idx_head_to = near_idx_head;   % default, we drive to next node
% cal the head to 
if abs(near_node_idx_best-near_node_idx_subopt) ~= 1
   
    if loc_dis_list_sort(near_idx_tail,2) / loc_dis_list_sort(near_idx_head,2) < 1/4 
        idx_head_to = near_idx_tail;
    else
        idx_head_to = near_idx_head;
    end
% if first head :TODO
end
% now already adjecent 
near_idx_tail = idx_head_to - 1;
%assert(near_idx_tail ~= 1);

if near_idx_tail <= 1
    flag_drv_directly = 1;
    idx_head_to = 1;
end



if idx_head_to > length(r_mark_x)
    idx_head_to = length(r_mark_x);
end

if idx_head_to < 1
    idx_head_to = 1;
    flag_drv_directly = 1;
end


if flag_drv_directly ~= 1
    % calculate degree of adjecent node  p0p1 p0p2
    % xyz_drv -> p0, tail -> p1, head -> p2
    % cal degree(p0) degree(p1), degree(p2)

    p0 = xyz_drv;
    p2 = road_mark(near_idx_head,:);
    p1 = road_mark(near_idx_tail,:);

    p3d_p0p1_p0p2_angle_ = p3d_v_angle_v(p0,p1,p2);
    
    p3d_p1p0_p1p2_angle_ = p3d_v_angle_v(p1,p0,p2);
    p3d_p2p1_p2p0_angle_ = p3d_v_angle_v(p2,p1,p0);
    
    
    if p3d_p1p0_p1p2_angle_>90
        idx_head_to = near_idx_tail;
        flag_need_divide_angle = 0;
        flag_drv_directly = 1;
    elseif p3d_p2p1_p2p0_angle_ > 90
            idx_head_to = near_idx_head + 1;
            flag_need_divide_angle = 0;
            flag_drv_directly = 1;
    else
        idx_head_to = near_idx_head;
        flag_need_divide_angle = 1;
        flag_drv_directly = 0;
    end
end


if idx_head_to > length(r_mark_x)
    idx_head_to = length(r_mark_x);
    flag_drv_directly = 1;
end

if idx_head_to < 1
    idx_head_to = 1;
    flag_drv_directly = 1;
end

if flag_drv_directly == 1
  dis_to_head = loc_dis_list_R0(idx_head_to,2);
  if dis_to_head > MAX_DEVIATION_DISTANCE 
      sprintf("at node, not normal line, DEVIATION at %d", idx_head_to)
      dis_loc_to_drv = road_mark(idx_head_to, :)
      idx_node_next = idx_head_to+1;
      if idx_node_next>length(road_mark)
          idx_node_next = length(road_mark);
      end
  end
elseif flag_need_divide_angle == 1 
    p0 = xyz_drv;
    %p1 = [r_mark_x(idx_head_to-1), r_mark_y(idx_head_to-1), r_mark_z(idx_head_to-1)];
    p1 = road_mark(idx_head_to-1,:);
    p2 = road_mark(idx_head_to  ,:);
    

    p3d_to_line_dis_ = p3d_to_line_dis(p0,p1,p2);
    dis_cal = sprintf("- count distance is : %0.3f", p3d_to_line_dis_);
    dis_cal;
    if p3d_to_line_dis_ > MAX_DEVIATION_DISTANCE
        % go normal line or normal head idx_head_to
        sprintf("NORMAL DEVIATION between %d and %d ", idx_head_to-1, idx_head_to)
        
        % cal go to direction
        l_tail = loc_dis_list_R0(idx_head_to-1,2);
        l_head = loc_dis_list_R0(idx_head_to  ,2);
        d = p3d_to_line_dis_;
        
        c = (d^2/cos(acos(d/l_head)/2)^2 + l_tail^2 - (2*d*l_tail*cos((pi*((90*acos(d/l_head))/pi + (180*acos(d/l_tail))/pi))/180))/cos(acos(d/l_head)/2))^(1/2);
        
        p1 = road_mark(idx_head_to-1,:);
        p2 = road_mark(idx_head_to  ,:);
        p1_2 = p1-p2;
        dis_p1_2 = sqrt( dot(p1_2,p1_2) );
        lambda = c / dis_p1_2;
        
        dis_loc_to_drv = (1-lambda)*p1 + lambda * p2;
        dis_loc_to_drv
        
        %road_mark(idx_head_to, :)
        
        if  if_p3d_p0_in_line_seg_p1_p2(dis_loc_to_drv, p1, p2)
            % pass
            idx_node_next = idx_head_to;
            flag_normal_deviation = 1;
        else
            flag_drv_directly = 1;
            dis_loc_to_drv = p2;
            idx_node_next = idx_head_to+1;
            if idx_node_next > len
                idx_node_next = len
            end
        end
        
        
        sprintf("- go to dis_loc_to_drv ")
        dis_loc_to_drv
        
        
    end
end


view([0,0,-1])
plot3(r_mark_x,r_mark_y,r_mark_z, '-O');
text(r_mark_x(near_idx_tail)-1, r_mark_y(near_idx_tail), r_mark_z(near_idx_tail), "(")
text(r_mark_x(near_idx_head)+1, r_mark_y(near_idx_head), r_mark_z(near_idx_head), ")")


text(r_mark_x(idx_head_to)-1, r_mark_y(idx_head_to), r_mark_z(idx_head_to), "->")




if ( dis_loc_to_drv ~= [0,0,0])  % deviation
    if flag_drv_directly 
    sprintf(" drv(%.2f,%.2f,%.2f) => node(%d) at (%.2f,%.2f,%.2f), ...-> node(%d)", xyz_drv(1),xyz_drv(2), xyz_drv(3), idx_head_to,  dis_loc_to_drv(1),dis_loc_to_drv(2), dis_loc_to_drv(3), idx_node_next)
    end
    
    if flag_normal_deviation
    sprintf("drv(%.2f,%.2f,%.2f) => (node(%d)~node(%d)) at (%.2f,%.2f,%.2f), ...-> node(%d)", xyz_drv(1),xyz_drv(2), xyz_drv(3), idx_head_to-1, idx_head_to,  dis_loc_to_drv(1),dis_loc_to_drv(2), dis_loc_to_drv(3), idx_node_next)
    end




    
    text(dis_loc_to_drv(1), dis_loc_to_drv(2),dis_loc_to_drv(3), "->>>");
    sprintf(" (%.2f,%.2f,%.2f) => (%.2f,%.2f,%.2f), -> node(%d)", xyz_drv(:,1),xyz_drv(:,2), xyz_drv(:,3),   dis_loc_to_drv(:,1),dis_loc_to_drv(:,2), dis_loc_to_drv(:,3), idx_node_next)
else
    "no need to turn"
    idx_node_next = idx_head_to;
    dis_loc_to_drv = road_mark(idx_node_next,:);
end


dis_loc_to_drv




axis equal;



hold on
plot3(x_drv, y_drv, z_drv, "-p")

xlabel("x");xticks(0:1:100);
ylabel("y");yticks(0:1:100);
zlabel("z");zticks(0:1:100);

for i=1:length(r_mark_x)
    text(r_mark_x(i),r_mark_y(i)+1,r_mark_z(i), int2str(i)); 
end

axis equal;
grid;
hold off;



function p3d_to_line_dis_ = p3d_to_line_dis(p0,p1,p2)
p3d_to_line_dis_ = norm(cross(p2'-p1',p0'-p1'))/norm(p2'-p1')
%p3d_to_line_dis_ = abs(det([p2-p1; p0-p1]))/norm(p2-p1);
end

function if_p3d_p0_in_line_seg_p1_p2_ = if_p3d_p0_in_line_seg_p1_p2(p0, p1,p2)
p_max= max(p1,p2);
p_min = min(p1,p2);
if_p3d_p0_in_line_seg_p1_p2_ = min( p_min <= p0 & p0 <= p_max );
end



function p3d_p0p1_p0p2_angle_ = p3d_v_angle_v(p0, p1,p2)
p0_1 = p1-p0;
p0_2 = p2-p0;
costheta = dot(p0_1,p0_2) / (sqrt( dot(p0_1,p0_1) ) * sqrt( dot(p0_2,p0_2) ));
theta = acosd(costheta);
p3d_p0p1_p0p2_angle_ = theta;
end




*********************
./uav_deviation_correct/src/test.m
*********************

% p3d_set = csvread("D:\\jd\\t\\3d.csv");
p3d_set = csvread("D:\\jd\\t\\lat_lon_h.csv"); 



% p3d_set = p3d_set ./ max(max(p3d_set))

plot3(p3d_set(:,1), p3d_set(:,2), p3d_set(:,3),'-*'); 

for i=1:length(p3d_set(:,1))
hold on;

% text(p3d_set(i,1)+1,p3d_set(i,2),p3d_set(i,3), int2str(i));

end

grid;
axis equal;
view([0,0,1]);




%syms x0 y0 z0 x1 y1 z1 x2 y2 z2;
% % 
% % %(x2,y2,z2);
% % 
% % 
% % 
% % A=[x0,y0,z0]; 
% % B=[x1,y1,z1];
% % %dot(A-B, B-C) == 0
% % %dot(A-B, A-B) == dot(B-C, B-C); 
% % %A=solve([(A-B).*(B-C) == 0,(A-B).*(A-B) == (B-C).*(B-C), z2==z1], [x2,y2]);
% 
% 
% A=solve([z2==z1, (x1-x0)*(x2-x1)+(y2-y1)*(y1-y0)+(z2-z1)*(z1-z0) == 0, (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0) == (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1) ],[x2,y2,z2]); 
% A
% 
% x0=41.5626;   
% y0=-0.0459;    
% z0=0.9871;
% 
% x1=55;
% y1=0;     
% z1=0;
% 
% 
% z2=z1;
% 
% x2_1=eval(A.x2(1));
% x2_2=eval(A.x2(2));
% y2_1=eval(A.y2(1));
% y2_2=eval(A.y2(2));
% 
% 
% 
% 
% A_ = [x0,y0,z0];
% B_ = [x1,y1,z1]; 
% C_1 = [x2_1, y2_1, z2];
% C_2 = [x2_2, y2_2, z2];
% 
% dot_ab_bc1 = dot(A_-B_ , B-C_1)
% dot_ab_bc2 = dot(A_-B_ , B_-C_2)
% 
% 
% line_len_ab = sqrt(dot(A_-B_, A_-B_))
% line_len_bc_1 = sqrt(dot(C_1-B_, C_1-B_))
% line_len_bc_2 = sqrt(dot(C_2-B_, C_2-B_))
% 
% A.x2
% A.y2



% 

% 
% a=[x0,y0,z0];
% b=[x1,y1,z1];
% c=[x2,y2,z2]
% 
% 
% 
% 
% 
% ans =
 
% (x0*x1 + y0*y1 - x1^2 - y1^2 - (y0*(x1^2*y1 - x1^2*y0 - 2*y0*y1^2 + y0^2*y1 + x0*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) - x1*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) + y1^3 + x0*x1*y0 - x0*x1*y1))/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2) + (y1*(x1^2*y1 - x1^2*y0 - 2*y0*y1^2 + y0^2*y1 + x0*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) - x1*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) + y1^3 + x0*x1*y0 - x0*x1*y1))/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2))/(x0 - x1)
% (x0*x1 + y0*y1 - x1^2 - y1^2 - (y0*(x1^2*y1 - x1^2*y0 - 2*y0*y1^2 + y0^2*y1 - x0*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) + x1*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) + y1^3 + x0*x1*y0 - x0*x1*y1))/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2) + (y1*(x1^2*y1 - x1^2*y0 - 2*y0*y1^2 + y0^2*y1 - x0*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) + x1*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) + y1^3 + x0*x1*y0 - x0*x1*y1))/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2))/(x0 - x1)
 

% 
%  x2=(x0*x1 + y0*y1 - x1^2 - y1^2 - (y0*(x1^2*y1 - x1^2*y0 - 2*y0*y1^2 + y0^2*y1 + x0*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) - x1*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) + y1^3 + x0*x1*y0 - x0*x1*y1))/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2) + (y1*(x1^2*y1 - x1^2*y0 - 2*y0*y1^2 + y0^2*y1 + x0*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) - x1*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) + y1^3 + x0*x1*y0 - x0*x1*y1))/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2))/(x0 - x1);
%  y2= (x1^2*y1 - x1^2*y0 - 2*y0*y1^2 + y0^2*y1 + x0*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) - x1*(- x0^4 + 2*x0^3*x1 - 2*x0^2*y0^2 + 2*x0^2*y0*y1 + x0^2*y1^2 - x0^2*z0^2 + x0^2*z1^2 - 2*x0*x1^3 + 2*x0*x1*y0^2 - 2*x0*x1*y0*y1 - 2*x0*x1*y1^2 + 2*x0*x1*z0^2 - 2*x0*x1*z1^2 + x1^4 + x1^2*y0^2 - 2*x1^2*y0*y1 + 2*x1^2*y1^2 - x1^2*z0^2 + x1^2*z1^2 - y0^4 + 2*y0^3*y1 - y0^2*z0^2 + y0^2*z1^2 - 2*y0*y1^3 + 2*y0*y1*z0^2 - 2*y0*y1*z1^2 + y1^4 - y1^2*z0^2 + y1^2*z1^2)^(1/2) + y1^3 + x0*x1*y0 - x0*x1*y1)/(x0^2 - 2*x0*x1 + x1^2 + y0^2 - 2*y0*y1 + y1^2);
%  z2=z1;
%  
% [x2,y2,z2]
% 
% C=[x2,y2,z2];
% 
% 
% axis equal;
%  plot3(x0,y0,z0,'*');
%   hold on;
%  plot3(x1,y1,z1,'O');
%   hold on;
%  plot3(x2,y2,z2,'p');
%  hold on;
%  
%  
%  
%  
%  
%  
%  





*********************
./uav_deviation_correct/src/uav_control.m
*********************



function [cur_loc_new, v_sum, dis_last_few, flag_arrived_dst,flag_deviation,...
    flag_arrived_to_verify_point,road_mark_verify_center, road_mark_prev, road_mark_verify] = uav_control(step_cnt, v_sum, flag_need_re_cal_v_sum, cur_loc, ...
    road_mark_prev, road_mark_next, speed_val, factor_ocean_d_uav, dis_last_few, dis_near_range, flag_deviation, max_n_dis_deviation, ...
    road_mark_verify, dis_verify_range)

if max( size(road_mark_prev) ) == 0
    road_mark_prev = cur_loc;
end

flag_arrived_dst = 0;
dt = 1;
factor_navig = 5/1000;

%gen_navig_dot_set_ = gen_navig_dot_set(cur_loc, road_mark_next, speed_val, dt, factor_navig);


%gen_navig_dot_set_

% grid;
% plot3(gen_navig_dot_set_(:,1), gen_navig_dot_set_(:,2), gen_navig_dot_set_(:,3), '-*');
% hold on;
% 
% view([0,0,1]);
% hold on;
% 
% axis equal;

if flag_need_re_cal_v_sum == 1 
v_sum = cal_v_sum(cur_loc, road_mark_next, speed_val, factor_ocean_d_uav);
end

cur_loc_new = cur_loc + v_sum * dt;




p0=cur_loc;

p1=road_mark_prev;
p2=road_mark_next;


[flag_arrived_dst, dis_last_few] = set_flag_arrived_rm_next(step_cnt, cur_loc, dis_last_few, dis_near_range, road_mark_prev, road_mark_next);

[flag_deviation] = set_flag_deviation(cur_loc, road_mark_prev, road_mark_next, max_n_dis_deviation);

[flag_arrived_to_verify_point, cur_loc, road_mark_verify_center, road_mark_verify] = set_flag_arrived_to_verify_point(cur_loc, road_mark_next, road_mark_verify, dis_verify_range);



end

function [flag_deviation] = set_flag_deviation(cur_loc, road_mark_prev, road_mark_next, max_n_dis_deviation)
flag_deviation = 0;
dis = p3d_to_line_n_dis(cur_loc,road_mark_prev,road_mark_next);
if dis > max_n_dis_deviation 
    flag_deviation = 1;
end



end

function [flag_arrived_to_verify_point, cur_loc, road_mark_verify_center, road_mark_verify] = set_flag_arrived_to_verify_point(cur_loc, road_mark_next, road_mark_verify, dis_verify_range)
flag_arrived_to_verify_point = 0;

if length(road_mark_verify) > 0
    
for i=1:length(road_mark_verify(:,1))
road_mark_verify_center = road_mark_verify(i,:);
dis = p3d_v_dis(road_mark_verify_center - cur_loc);
%dis
if dis < dis_verify_range
    "- enter verify area"
   
    flag_arrived_to_verify_point=1;
    road_mark_verify(i,:) = []; % clear this 3d verify point
    break;
end




end
else
   road_mark_verify_center = []; 
end




end


function [flag_arrived_dst,dis_last_few] = set_flag_arrived_rm_next(step_cnt, cur_loc, dis_last_few, dis_near_range, road_mark_prev, road_mark_next)
flag_arrived_dst = 0;


if_p3d_proj_p_in_line_seg_ = if_p3d_proj_p_in_line_seg(cur_loc,road_mark_prev,road_mark_next);
if if_p3d_proj_p_in_line_seg_ ~= 1
    flag_arrived_dst = 1;
else 
    sz_dis_last_few = length(dis_last_few);
    idx_dis = mod(step_cnt-1, sz_dis_last_few)+1;


    dis_to_dst = p3d_v_dis(cur_loc-road_mark_next);
    dis_last_few(idx_dis) = dis_to_dst;

    med_dis = median(dis_last_few);
    if med_dis < dis_near_range
        "arrived"
        flag_arrived_dst = 1;
    end

end
if flag_arrived_dst == 1
    dis_last_few = dis_last_few.*0 + dis_near_range;
end


end



function cal_v_sum_ = cal_v_sum(cur_loc, road_mark_next, speed_val, factor_ocean_d_uav)
v_uav_speed_direct = road_mark_next - cur_loc; 

v_uav_speed_unit = v_to_unit(v_uav_speed_direct); 
v_ocean_unit = get_v_ocean_unit();

v_uav_speed = v_uav_speed_unit * get_uav_speed(speed_val);
v_ocean_speed = v_ocean_unit * get_uav_speed(speed_val) * factor_ocean_d_uav;

v_sum = v_uav_speed + v_ocean_speed;
cal_v_sum_ = v_sum;
end



function p3d_to_line_dis_ = p3d_to_line_n_dis(p0,p1,p2)
p3d_to_line_dis_ = norm(cross(p2'-p1',p0'-p1'))/norm(p2'-p1');
%p3d_to_line_dis_ = abs(det([p2-p1; p0-p1]))/norm(p2-p1);
end


function if_p3d_proj_p_in_line_seg_ = if_p3d_proj_p_in_line_seg(p0,p1,p2)
if_p3d_proj_p_in_line_seg_ = 0;
v_p1_p0=p0-p1;
v_p1_p2 = p2-p1;
v_p0_p2 = p2-p0;

cos_theta_prev = dot(v_p1_p0, v_p1_p2);
cos_theta_next = dot(v_p1_p2, v_p0_p2);

if cos_theta_prev >=0 && cos_theta_next>=0
    if_p3d_proj_p_in_line_seg_ = 1;
end

if_p3d_proj_p_in_line_seg_;


end



function v_to_unit_ = v_to_unit(v)
v_to_unit_ = v .* (1/p3d_v_dis(v));
end


function p3d_v_dis_ = p3d_v_dis(v)
p3d_v_dis_ = sqrt(dot(v,v));
end

function get_uav_speed_ = get_uav_speed(speed)
get_uav_speed_ = speed;
end

function get_v_ocean_unit_ = get_v_ocean_unit()
% v_o = [randi(10)-5, randi(10)-5, randi(10)/10];
v_o = [-5, +5, 0.001]; % later edit this to random ocean current
v_o_unit = v_to_unit(v_o);
get_v_ocean_unit_ = v_o_unit;
end
function gen_rnd_unit_ = gen_rnd_unit()
v = [randi(10)-6, randi(10)-6,randi(10)-6+0.02];
v_unit = v_to_unit(v);
gen_rnd_unit_ = v_unit;
end

function gen_navig_dot_set_ = gen_navig_dot_set(start_p, end_p, speed_val, dt, factor)
% if speed change, need re-calculate this navig dot set
v = end_p - start_p;
dis = p3d_v_dis(v);
err_radius = dis * factor;
err_v = gen_rnd_unit().*err_radius;
navig_num = dis/speed_val/dt;  % step_cnt
e_dis = dis/navig_num;

cnt= 1;
gen_navig_dot_set_(cnt,:) = start_p;
cnt = cnt+1;
init_v_drv = end_p-start_p;  % never change direction from now

for i = 1:navig_num
    e_err_radius = e_dis*factor*1;  % based on last loc
    e_err_v = gen_rnd_unit()*e_err_radius;
    e_center_loc = start_p + init_v_drv.*(1/navig_num);  % approximation
    start_p_new = e_center_loc + e_err_v;
    start_p = start_p_new;
    gen_navig_dot_set_(cnt,:) = start_p;
    cnt = cnt+1;
end





end


*********************
./uav_deviation_correct/src/v_ship_v_ocean_mix.m
*********************
function v_ship_v_ocean_mix_ = v_ship_v_ocean_mix(loc_ship , v_ship_t0, v_ocean_t0)

%v_ship_v_ocean_mix_;
v_ship_v_ocean_mix_  = zeros(100,3);
for i = 1:100
    dt =1 * i;
    
    v_all = v_ship_t0 + v_ocean_t0; % t0
    
    v_all_dt = v_all * dt; %
    
    v_ship_v_ocean_mix_(i,:) = loc_ship  + v_all_dt; %t1
end

end




*********************
./Â£∞Â≠¶ÈÄÇÈÖçÂå∫ÁîüÊàêÁÆóÊ≥ïÊ®°Âûã/Â£∞Â≠¶ÈÄÇÈÖçÂå∫ÁîüÊàêÁÆóÊ≥ïÊ®°Âûã/Acousitc_area_f.m
*********************
function [SNR,SNR2,XX,XX2,Fugai,NR,NHR]=Acousitc_area_f(Aca_inp)
SNR_mode = Aca_inp.SNR_mode;
XX_mode = Aca_inp.XX_mode;
SL = Aca_inp.SL;
Work_Depth=Aca_inp.Work_Depth;
F0_inp=Aca_inp.F0;
SourceZ = Aca_inp.SourceZ;
ssp = Aca_inp.ssp;
T_inp = Aca_inp.T;
SNR_menxian_inp = Aca_inp.SNR_menxian;
XX_menxian = Aca_inp.XX_menxian;
w =Aca_inp.w;
NL_mode = Aca_inp.NL_mode;
SourceZ = Aca_inp.SourceZ;
TT = Aca_inp.TT;
L = Aca_inp.L;
SNR_Ang = Aca_inp.SNR_Ang;
%% ∏≤∏«∑∂Œßº∆À„
if SNR_mode==1
    %       [SNR,SNR2,Fugai,NR,NHR] = Use_BHP_f(SL,Work_Depth,F0,SourceZ,ssp,T,SNR_menxian,w,NL_mode);
else
    [SNR,SNR2,Fugai,NR,NHR] = Use_TLmodel_f(SL,Work_Depth,F0_inp,SourceZ,ssp,T_inp,SNR_menxian_inp,w,NL_mode,SNR_Ang);
end

%% ◊˜”√æ´∂»º∆À„
if XX_mode==1
    c = mean(ssp(:,2));
    [XX,XX2,R_all,R2] = Use_USBL_f(SNR,SNR2,SourceZ,Fugai,NR,NHR,w,TT,F0_inp,L,c);
else
    %     [XX,XX2,R_all,R2] = Use_CorrR_f(SNR,SourceZ,Fugai,NR,NHR,XX_menxian);
end

end




function [SNR,SNR2,Fugai,NR,NHR] = Use_TLmodel_f(SL,Work_Depth,F0_inp,SourceZ,ssp,T_inp,SNR_menxian_inp,w,NL_mode,SNR_Ang)

%% ≤Œ ˝≈‰÷√
%ª˘◊º…Ó∂»
SD = SourceZ;
%Ω” ’…˘‘¥Œª÷√
if SD>1000
    RH = 0:200:20000;
    RD = 0:10:(round(SD/10))*10;
else
    if SD>100
        RH = 0:100:15000;
        RD = 0:5:(round(SD/10))*10;
    else
        RH = 0:50:10000;
        RD = 0:1:round(SD);
    end
end

%Œ¸ ’À ß≤Œ ˝
T0 = T_inp;
F0 = F0_inp/1000;
SNR_menxian = SNR_menxian_inp;
%% %%%%%%%%%%%%% ¥´≤•À ßº∆À„%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Œ¸ ’À ßº∆À„
T = [5 10 15];
F = [0.5 1 2 5 10 20 50 100 200 500];
alpha = [0.02 0.06 0.14 0.33 1 3.8 15 30 55 120;
    0.02 0.06 0.14 0.29 0.82 3.3 16 35 60 125;
    0.02 0.06 0.14 0.26 0.68 2.8 17 40 65 130];

map_out = zeros(length(RD),length(RH));
for i=1:length(RD)
    map_out0 = 1;
    for j = 1:length(RH)
        R(i,j) = sqrt( (RD(i)-SD)^2 + RH(j).^2 );
        Dr = RH(2)-RH(1);
        
        if map_out0==1
            if RD(i)<SD
                map_out(i,j) = X_to_alpha(ssp(:,1),ssp(:,2),RD(i),SD,R(i,j),Dr);
            else
                map_out(i,j) = X_to_alpha(ssp(:,1),ssp(:,2),SD,RD(i),R(i,j),Dr);
            end
            if isnan(map_out(i,j))||(asind(RH(j)/R(i,j))>SNR_Ang)
                map_out0=0;
            end
        else
            map_out(i,j)=nan;
        end
        
        
        if F0 == F
            F_equ_num = find(F0 == F);
            if T0 == T      %…Ë÷√∂‘”¶∆µ¬ ∫ÕŒ¬∂»µ»Õ¨ ±
                T_equ_num = find(T0 == T);
                alpha_out = alpha(T_equ_num,F_equ_num) ;
            else            %…Ë÷√∂‘”¶∆µ¬ œ‡Õ¨∫ÕŒ¬∂»≤ªÕ¨ ±
                alpha0 = alpha(:,F_equ_num);
                alpha_out = interp1(T,alpha0,T0,'linear','extrap');
            end
        else
            if T0 == T      %…Ë÷√∂‘”¶∆µ¬ ≤ªÕ¨£¨Œ¬∂»œ‡Õ¨ ±
                T_equ_num = find(T0 == T);
                alpha0 = alpha(T_equ_num,:) ;
                alpha_out = interp1(F,alpha0,F0,'linear','extrap');
            else            %…Ë÷√∂‘”¶∆µ¬ ∫ÕŒ¬∂»æ˘≤ªÕ¨ ±
                %œ»≤Â∆µ¬ 
                for k = 1:3
                    alpha0(k) = interp1(F,alpha(k,:),F0,'linear','extrap');
                end
                %‘Ÿ≤ÂŒ¬∂»
                alpha_out = interp1(T,alpha0,T0,'linear','extrap');
            end
        end
        alpha_out;
        
        %% ¿©’πÀ ßº∆À„
        r= R(i,j);
        TL(i,j) =  20*log10(r) + alpha_out*r/1000;
    end
end
%% NLº∆À„
F = [0.5 1 2 4 8 16 32 64];
NL0 = [75 70 65 60 55 50 45 40;
    65 60 55 50 45 40 35 30;
    84 78 72 66 60 54 48 42;
    74 68 62 60 50 44 38 32];
if F0 == F
    mode_num = NL_mode;
    F_equ_num = find(F0 == F);
    NL0_out = NL0(mode_num,F_equ_num) ;
else
    mode_num = NL_mode;
    NL0_wait = NL0(mode_num,:);
    NL0_out = interp1(F,NL0_wait,F0,'linear','extrap');
end

SNR = SL - TL - NL0_out - 10*log10(w);
SNR = SNR.*map_out;
SNR(SNR<SNR_menxian)=nan;
Range = (find(SNR(Work_Depth/(RD(2)-RD(1)),:)>SNR_menxian,1,'last'))*(RH(2)-RH(1));
SNR2 = SNR(Work_Depth/(RD(2)-RD(1)),:);
Fugai = [Range/1000 Work_Depth];


NR =  RD;
NHR = RH./1000;
% figure;
%     h = pcolor( RH/1000, RD, SNR );
%     shading flat;
%     colorbar
%     set( gca, 'YDir', 'Reverse' )
%     tej = flipud( jet( 256 ) );
%     colormap( tej );
% %     colormap( tej(1:25:251,:) );
%     hold on
%     plot([0 Fugai(1)],[Fugai(2) Fugai(2)],'k','linewidth',2);
%     hold on
%     plot(Fugai(1),Fugai(2),'.k','markersize',20);
%     xlabel('Range (km)' )
%     ylabel( 'Depth (m)' );
%     legend('TL','R-Depth')
%     set(gca,'FontName','Times New Roman','FontSize',17,'LineWidth',1.5);set(gcf,'position',[100,100,1000,450])
%

end


%% …˘œﬂ∏˙◊Ÿ∑®◊”∫Ø ˝≥Ã–Ú
function map_out = X_to_alpha(depth,shengsu,z_source,z_receiver,x_horz,Dr)
if z_receiver > depth(end)
    error('…˘ÀŸ∆ √Ê…Ó∂»≤ª◊„£°')
end
if x_horz ==0
    theta = 90;
else
    for numnum = 1:10000
        if numnum>10
            map_out = nan;
        end
        if numnum == 1
            theta = 45;
            theta1 = theta;
        elseif x - x_horz > 0
            theta = theta + 45/(2^(numnum-1));
            theta1 = theta;
        elseif x - x_horz < 0
            theta = theta - 45/(2^(numnum-1));
            theta1 = theta;
        end
        x = 0;
        if theta1 == 90
            x = 0;
        else
            for m = 1:length(depth)
                if z_source-depth(1) == 0
                    num1 = 1;
                    break;
                elseif z_source - depth(m) < 0
                    num1 = m-1;
                    break;
                end
            end
            for m = 1:length(depth)
                if z_receiver - depth(m) <= 0
                    num2 = m - 1;
                    break
                end
            end
            
            for n = num1:num2
                a = (shengsu(n+1)-shengsu(n))/((depth(n+1)-depth(n))*shengsu(n));
                if n == num1
                    z0 = z_source;
                    c0 = shengsu(n)*(1+a*(z0-depth(n)));
                else
                    z0 = depth(n);
                    c0 = shengsu(n);
                end
                if n == num2
                    z1 = z_receiver;
                    c1 = shengsu(n-1)*(1+a*(z1-z0));
                else
                    z1 = depth(n+1);
                    c1 = shengsu(n+1);
                end
                theta0 = theta1;
                theta1=acosd(c1/c0*cosd(theta0));
                x1=abs(z1-z0)/tand((theta0+theta1)/2);
                x=x1+x;
            end
        end
        if abs(x-x_horz) <= Dr
            map_out = 1;
            break;
        end
        
    end
end
end

function [XX,XX2,R_all,R2] = Use_USBL_f(SNR,SNR2,SourceZ,Fugai,NR,NHR,w,TT,F0,L,c)

%% º∆À„ ±—”π¿º∆æ´∂»
dD = sqrt(1/(4*(pi^2))) * (1./sqrt(db2pow(SNR))) * ...
    (1./sqrt(w*TT)) ./F0 ./sqrt(1+(w.^2)/(12*(c.^2)));   %∏ﬂ–≈‘Î±»

% dD = sqrt(1/(8*(pi^2))) * (1./(db2pow(SNR))) * ...
%     (1./sqrt(w*TT)) ./F0 ./sqrt(1+(w.^2)/(12*(F0.^2)));      %µÕ–≈‘Î±»

dD2 = sqrt(1/(4*(pi^2))) * (1./sqrt(db2pow(SNR2))) * ...
    (1./sqrt(w*TT)) ./F0 ./sqrt(1+(w.^2)/(12*(c.^2)));   %∏ﬂ–≈‘Î±»

% dD2 = sqrt(1/(8*(pi^2))) * (1./(db2pow(SNR))) * ...
%     (1./sqrt(w*TT)) ./F0 ./sqrt(1+(w.^2)/(12*(F0.^2)));      %µÕ–≈‘Î±»

%% º∆À„∂®ŒªŒÛ≤Ó
for i=1:length(NR)
    for j = 1:length(NHR)
        R_all(i,j) = sqrt( (NR(i)-SourceZ)^2 + (NHR(j).*1000).^2 );
    end
end
XX = sqrt(2*((c/L)^2)*((sqrt(2).*dD).^2)).*R_all;
HR2 = NHR(1):NHR(2)-NHR(1):Fugai(1);
R2 = sqrt( (Fugai(2)-SourceZ).^2 + (HR2.*1000).^2 );
XX2 = sqrt(2*((c/L)^2)*((sqrt(2).*dD2(1:length(HR2))).^2)).*R2;

% figure;
%     h = pcolor( NHR, NR, XX );
%     shading flat;
%     colorbar
%     set( gca, 'YDir', 'Reverse' )
%     tej = flipud( jet( 256 ) );
%     colormap( tej );
% %     colormap( tej(1:25:251,:) );
%         hold on
%     plot([0 Fugai(1)],[Fugai(2) Fugai(2)],'k','linewidth',2);
%     hold on
%     plot(Fugai(1),Fugai(2),'.k','markersize',20);
%     xlabel('Range (km)' )
%     ylabel( 'Depth (m)' );
%     set(gca,'FontName','Times New Roman','FontSize',17,'LineWidth',1.5);set(gcf,'position',[100,100,1000,450])
%
%     figure;
%     plot(HR2,XX2,'k','linewidth',2);grid minor;
%     xlabel('Range (km)' )
%     ylabel( 'Depth (m)' );
%     set(gca,'FontName','Times New Roman','FontSize',17,'LineWidth',1.5);set(gcf,'position',[100,100,1000,450])
end
*********************
./Â£∞Â≠¶ÈÄÇÈÖçÂå∫ÁîüÊàêÁÆóÊ≥ïÊ®°Âûã/Â£∞Â≠¶ÈÄÇÈÖçÂå∫ÁîüÊàêÁÆóÊ≥ïÊ®°Âûã/cal_h_theta_loc.m
*********************
syms l_tail l_head d ;


theta = acosd(d/l_head);
l_m = d/cosd(theta/2);
r = acosd(d/l_tail);

% l_tail = 1; l_head=1; d = 1/sqrt(2) ;
c = sqrt( l_tail^2 + l_m^2 -2*l_tail*l_m*cosd(r+theta/2) );
c
% (d^2/cos(acos(d/l_head)/2)^2 + l_tail^2 - (2*d*l_tail*cos((pi*((90*acos(d/l_head))/pi + (180*acos(d/l_tail))/pi))/180))/cos(acos(d/l_head)/2))^(1/2)
*********************
./Â£∞Â≠¶ÈÄÇÈÖçÂå∫ÁîüÊàêÁÆóÊ≥ïÊ®°Âûã/Â£∞Â≠¶ÈÄÇÈÖçÂå∫ÁîüÊàêÁÆóÊ≥ïÊ®°Âûã/corr_direct.m
*********************


MAX_DEVIATION_DISTANCE = 3

dis_loc_to_drv = [0,0,0]
idx_node_next = 0;


flag_drv_directly = 0;
flag_normal_deviation = 0;

%x = round(sort(rand(10,1))*77); y = round(rand(10,1)*10); z = round(rand(10,1)*4); csvwrite("p3d.csv", [x,y,z]); 
%t = round(sort(randi(300,200,1)));tu = unique(t); x=tu(1:100); y = round(rand(100,1)*10);z = round(rand(100,1)*4); csvwrite("p3d_100.csv", [x,y,z])
grid;
road_mark = csvread("p3d.csv");

r_mark_x = road_mark(:,1);
r_mark_y = road_mark(:,2);
r_mark_z = road_mark(:,3);

idx_rnd = randperm(length(r_mark_x)-1);
x_drv = r_mark_x(idx_rnd(1)) + rand(1)* 20;
y_drv = r_mark_y(idx_rnd(1)) + rand(1)* 8;
z_drv = r_mark_z(idx_rnd(1)) + rand(1)* 4;
xyz_drv = [x_drv, y_drv, z_drv];

if 1
    xyz_drv = [48.106409724510650,8.246225127798446,2.909241326569151];
    x_drv = xyz_drv(1);
    y_drv = xyz_drv(2);
    z_drv = xyz_drv(3);
    % "drv(48.11,8.25,2.91) => (node(8)~node(9)) at (47.48,3.89,2.37), ...-> node(9)"
end


xyz_drv


dis_list = sqrt((r_mark_x-x_drv).^2 +  (r_mark_y-y_drv).^2 + (r_mark_z-z_drv).^2);
loc_dis_list = [ (1:length(r_mark_x))', dis_list];


loc_dis_list;

loc_dis_list_R0 = loc_dis_list;

loc_dis_list_sort = sortrows(loc_dis_list, 2);
%loc_dis_list_sort





near_node_idx_best = loc_dis_list_sort(1,1);
near_node_idx_subopt = loc_dis_list_sort(2,1);

near_idx_head = max(near_node_idx_best, near_node_idx_subopt);
near_idx_tail = min(near_node_idx_best, near_node_idx_subopt);

% if the two nodes is adjacent ? 
flag_need_divide_angle = 1;
idx_head_to = near_idx_head;   % default, we drive to next node
% cal the head to 
if abs(near_node_idx_best-near_node_idx_subopt) ~= 1
   
    if loc_dis_list_sort(near_idx_tail,2) / loc_dis_list_sort(near_idx_head,2) < 1/4 
        idx_head_to = near_idx_tail;
    else
        idx_head_to = near_idx_head;
    end
% if first head :TODO
end
% now already adjecent 
near_idx_tail = idx_head_to - 1;
assert(near_idx_tail ~= 1);

if near_idx_tail <= 1
    flag_drv_directly = 1;
    idx_head_to = 1;
end



if idx_head_to > length(r_mark_x)
    idx_head_to = length(r_mark_x);
end

if idx_head_to < 1
    idx_head_to = 1;
    flag_drv_directly = 1;
end


if flag_drv_directly ~= 1
    % calculate degree of adjecent node  p0p1 p0p2
    % xyz_drv -> p0, tail -> p1, head -> p2
    % cal degree(p0) degree(p1), degree(p2)

    p0 = xyz_drv;
    p2 = road_mark(near_idx_head,:);
    p1 = road_mark(near_idx_tail,:);

    p3d_p0p1_p0p2_angle_ = p3d_v_angle_v(p0,p1,p2);
    
    p3d_p1p0_p1p2_angle_ = p3d_v_angle_v(p1,p0,p2);
    p3d_p2p1_p2p0_angle_ = p3d_v_angle_v(p2,p1,p0);
    
    
    if p3d_p1p0_p1p2_angle_>90
        idx_head_to = near_idx_tail;
        flag_need_divide_angle = 0;
        flag_drv_directly = 1;
    elseif p3d_p2p1_p2p0_angle_ > 90
            idx_head_to = near_idx_head + 1;
            flag_need_divide_angle = 0;
            flag_drv_directly = 1;
    else
        idx_head_to = near_idx_head;
        flag_need_divide_angle = 1;
        flag_drv_directly = 0;
    end
end


if idx_head_to > length(r_mark_x)
    idx_head_to = length(r_mark_x);
    flag_drv_directly = 1;
end

if idx_head_to < 1
    idx_head_to = 1;
    flag_drv_directly = 1;
end

if flag_drv_directly == 1
  dis_to_head = loc_dis_list_R0(idx_head_to,2);
  if dis_to_head > MAX_DEVIATION_DISTANCE 
      sprintf("at node, not normal line, DEVIATION at %d", idx_head_to)
      dis_loc_to_drv = road_mark(idx_head_to, :)
      idx_node_next = idx_head_to+1;
      if idx_node_next>length(road_mark)
          idx_node_next = length(road_mark);
      end
  end
elseif flag_need_divide_angle == 1 
    p0 = xyz_drv;
    %p1 = [r_mark_x(idx_head_to-1), r_mark_y(idx_head_to-1), r_mark_z(idx_head_to-1)];
    p1 = road_mark(idx_head_to-1,:);
    p2 = road_mark(idx_head_to  ,:);
    

    p3d_to_line_dis_ = p3d_to_line_dis(p0,p1,p2);
    dis_cal = sprintf("- count distance is : %0.3f", p3d_to_line_dis_);
    dis_cal;
    if p3d_to_line_dis_ > MAX_DEVIATION_DISTANCE
        % go normal line or normal head idx_head_to
        sprintf("NORMAL DEVIATION between %d and %d ", idx_head_to-1, idx_head_to)
        
        % cal go to direction
        l_tail = loc_dis_list_R0(idx_head_to-1,2);
        l_head = loc_dis_list_R0(idx_head_to  ,2);
        d = p3d_to_line_dis_;
        
        c = (d^2/cos(acos(d/l_head)/2)^2 + l_tail^2 - (2*d*l_tail*cos((pi*((90*acos(d/l_head))/pi + (180*acos(d/l_tail))/pi))/180))/cos(acos(d/l_head)/2))^(1/2);
        
        p1 = road_mark(idx_head_to-1,:);
        p2 = road_mark(idx_head_to  ,:);
        p1_2 = p1-p2;
        dis_p1_2 = sqrt( dot(p1_2,p1_2) );
        lambda = c / dis_p1_2;
        
        dis_loc_to_drv = (1-lambda)*p1 + lambda * p2;
        dis_loc_to_drv
        
        %road_mark(idx_head_to, :)
        
        if  if_p3d_p0_in_line_seg_p1_p2(dis_loc_to_drv, p1, p2)
            % pass
            idx_node_next = idx_head_to;
            flag_normal_deviation = 1;
        else
            flag_drv_directly = 1;
            dis_loc_to_drv = p2;
            idx_node_next = idx_head_to+1;
            if idx_node_next > len
                idx_node_next = len
            end
        end
        
        
        sprintf("- go to dis_loc_to_drv ")
        dis_loc_to_drv
        
        
    end
end


view([0,0,-1])
plot3(r_mark_x,r_mark_y,r_mark_z, '-O');
text(r_mark_x(near_idx_tail)-1, r_mark_y(near_idx_tail), r_mark_z(near_idx_tail), "(")
text(r_mark_x(near_idx_head)+1, r_mark_y(near_idx_head), r_mark_z(near_idx_head), ")")


text(r_mark_x(idx_head_to)-1, r_mark_y(idx_head_to), r_mark_z(idx_head_to), "->")




if ( dis_loc_to_drv ~= [0,0,0])  % deviation
    if flag_drv_directly 
    sprintf(" drv(%.2f,%.2f,%.2f) => node(%d) at (%.2f,%.2f,%.2f), ...-> node(%d)", xyz_drv(1),xyz_drv(2), xyz_drv(3), idx_head_to,  dis_loc_to_drv(1),dis_loc_to_drv(2), dis_loc_to_drv(3), idx_node_next)
    end
    
    if flag_normal_deviation
    sprintf("drv(%.2f,%.2f,%.2f) => (node(%d)~node(%d)) at (%.2f,%.2f,%.2f), ...-> node(%d)", xyz_drv(1),xyz_drv(2), xyz_drv(3), idx_head_to-1, idx_head_to,  dis_loc_to_drv(1),dis_loc_to_drv(2), dis_loc_to_drv(3), idx_node_next)
    end




    
    text(dis_loc_to_drv(1), dis_loc_to_drv(2),dis_loc_to_drv(3), "->>>");
    sprintf(" (%.2f,%.2f,%.2f) => (%.2f,%.2f,%.2f), -> node(%d)", xyz_drv(:,1),xyz_drv(:,2), xyz_drv(:,3),   dis_loc_to_drv(:,1),dis_loc_to_drv(:,2), dis_loc_to_drv(:,3), idx_node_next)
else
    "no need to turn"
end





axis equal;



hold on
plot3(x_drv, y_drv, z_drv, "-p")

xlabel("x");xticks(0:1:100);
ylabel("y");yticks(0:1:100);
zlabel("z");zticks(0:1:100);

for i=1:length(r_mark_x)
    text(r_mark_x(i),r_mark_y(i)+1,r_mark_z(i), int2str(i)); 
end

axis equal;
grid;
hold off;



function p3d_to_line_dis_ = p3d_to_line_dis(p0,p1,p2)
p3d_to_line_dis_ = norm(cross(p2'-p1',p0'-p1'))/norm(p2'-p1')
%p3d_to_line_dis_ = abs(det([p2-p1; p0-p1]))/norm(p2-p1);
end

function if_p3d_p0_in_line_seg_p1_p2_ = if_p3d_p0_in_line_seg_p1_p2(p0, p1,p2)
p_max= max(p1,p2);
p_min = min(p1,p2);
if_p3d_p0_in_line_seg_p1_p2_ = min( p_min <= p0 & p0 <= p_max );
end



function p3d_p0p1_p0p2_angle_ = p3d_v_angle_v(p0, p1,p2)
p0_1 = p1-p0;
p0_2 = p2-p0;
costheta = dot(p0_1,p0_2) / (sqrt( dot(p0_1,p0_1) ) * sqrt( dot(p0_2,p0_2) ));
theta = acosd(costheta);
p3d_p0p1_p0p2_angle_ = theta;
end




*********************
./Â£∞Â≠¶ÈÄÇÈÖçÂå∫ÁîüÊàêÁÆóÊ≥ïÊ®°Âûã/Â£∞Â≠¶ÈÄÇÈÖçÂå∫ÁîüÊàêÁÆóÊ≥ïÊ®°Âûã/main_test.m
*********************
%% …˘—ß  ≈‰«¯∑÷Œˆ
%% ±‡º≠£∫ccy
%% »’∆⁄£∫2022/10/19

clc
clear all
close all

%% ≤Œ ˝input
Aca_inp.SNR_mode = 2;  %1Œ™bellhop(ø’)   2Œ™¥´≤•À ß
Aca_inp.XX_mode = 1;   %1Œ™¿Ì¬€π´ Ω  2Œ™æ≠—È÷µ(ø’) 
%∆ΩÃ®ƒ£–Õ
Aca_inp.Work_Depth = 300;    %∆ΩÃ®Ω” ’…Ó∂»
%ª˘◊ºƒ£–Õ
Aca_inp.F0=3000;             % –≈∫≈∆µ¬ £¨µ•Œª£∫Hz
Aca_inp.SourceZ=1995;         % …˘‘¥…Ó∂»£¨µ•Œª£∫m
Aca_inp.BLH = [];           %ª˘◊ºæ≠Œ≥∏ﬂ
Aca_inp.SL = 185;   %…˘‘¥º∂
Aca_inp.w = 2000;   %–≈∫≈¥¯øÌ
Aca_inp.TT = 0.3;    %–≈∫≈¬ˆøÌ
Aca_inp.SNR_Ang = 80  ; %◊∂–Œø™Ω«
%USBLƒ£–Õ
Aca_inp.SNR_menxian = 0;
Aca_inp.XX_menxian = [60 0.003;...
               80 0.005;...
               90 0.01];
Aca_inp.L=0.26;

%À ßƒ£–Õ
% Aca_inp.F0 = F0;
Aca_inp.T = 5;
Aca_inp.NL_mode = 3;

%∫£—Ûª∑æ≥
load ƒœ∫£2000…Ó12‘¬E119.00N21.40Span0.3.mat     %CTD,
ssp0 = ssp;
Aca_inp.ssp =[ssp0(1:10:end,:)];

%≤‚ªÊµÿ¿Ì
%load µÿ–Œµ◊÷ 
Aca_inp.ocean_top = []; %µÿ–Œµ◊÷ 

[SNR,SNR2,XX,XX2,Fugai,NR,NHR]=Acousitc_area_f(Aca_inp);
EOF


%%  matlab class example
classdef xyz

    
    properties
        x
        y
    end
    
    methods
        function obj = xyz(x_,y_)

            obj.x = x_;
            obj.y = y_;
        end
        
        function s_ = see_xyz(obj)

            s_ = disp(obj.x, obj.y);
        end
        function obj2 = add(obj, obj1)
            obj2.x = obj.x + obj1.x;
            obj2.y = obj.y + obj1.y; 
        end
    end
end


 
 
