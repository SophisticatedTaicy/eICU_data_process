%获取patient表数据
patient = readtable('apache4.csv');
%提取所有age数据（此时为tabel类型）
apache = patient(:,'apache4');
%类型转换为数组
apache=table2array(apache);
%对于数组中>89的数据会显示为nan,将这部分值转换为90
apache(isnan(apache)==1)=0;
y = 0:1:220;
%计算age的方差和均值
[mu,sigma]= normfit(apache);
%normal最终输出数组的正太分布图
x = pdf('Normal',y,mu,sigma);
xlabel('Estimated density')
ylabel('APACHE IV score ad admission')
plot(x,y);


