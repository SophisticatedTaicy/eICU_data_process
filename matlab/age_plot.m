%获取patient表数据
patient = readtable('patient.csv');
%提取所有age数据（此时为tabel类型）
age = patient(:,'age');
%类型转换为数组
age=table2array(age);
%对于数组中>89的数据会显示为nan,将这部分值转换为90
age(isnan(age)==1)=90;
y = 10:0.1:100;
%计算age的方差和均值
[mu,sigma]= normfit(age);
%normal最终输出数组的正太分布图
x = pdf('Normal',y,mu,sigma);
xlabel('age at admission');
ylabel('estimate density');
plot(x,y);
%设置x,y轴名称
%[x,y]=ksdensity(age);
set(gca,'XTick',[10 20 40 60 80 100]);
set(gca,'YTick',[0.01 0.02]);
%绘制图表
plot(x,y)
