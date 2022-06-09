[TOC]

# 一.基本用法

## 1. 帮助命令

1. help 命令：简单介绍
2. doc 命令：网页文档
3. lookfor 命令：模糊查找

## 2. 数据的输入

### 2.1矩阵的输入

```matlab
a=[1 2 3; 4 5 6; 7 8 9]
```

### 2.2 特殊变量

```matlab
ans %用于结果的缺省变量名
pi %圆周率
eps %浮点相对精度
inf %无穷大 1/0
NaN %不定量 如0/0
i(j) %i=j=虚数
nargin %所有函数的输入变量的数目
nargout %所有函数的输出变量的数目
realmin %最小正实数
realmax %最大正实数
```

### 2.3 特殊向量和特殊矩阵

特殊向量

```matlab
t=[0:0.1:10] %从0到10的行向量，元素间隔0.1
t=linspace(始,末,[变量个数,默认100]) %从始到末的线性分布的n个数
t=logspace(n1,n2,[n,默认50])%在10^n1和10^n2之间按照对数等间距产生的n个数
```

特殊矩阵

```matlab
eyes() %单位矩阵
ones() %全1矩阵
zeros()%全0矩阵
q=[] %空矩阵.作用1：初始化，作用2：删除矩阵的部分，作用3：函数参数取默认值
reshape(1:16,[4,4])%产生1:14
%随机数
rand() %服从[0,1]上均匀分布的随机矩阵
normrnd(mu,sigma,m,n)%产生mxn的矩阵，均值mu，标准差sigma，正态分布
expnd(mu,m,n)%均值mu，指数分布
poissrnd(mu,m,n)%均值mu，泊松分布
unifrnd(a,b,m,n)%在区间[a,b]均匀分布
%随机置换
randperm(n)%产生1到n的一个随机全排列
perms([1,n])%产生1到n的所有全排列
```

特殊用法

```matlab
d3(:)%按列展开
repmat(x,4,1)%把行向量x赋值4行1列
```



## 3. 矩阵的四则运算

### 3.1 加减乘除、幂运算、逆阵

加(+)减(-)乘(*)除(左广义逆/，右广义逆\\)、幂运算(^)、逆阵(inv)

求余数`rem(10,4)=2`

### 3.2 逐位运算

`.*`, `./`, `.\`, `.^`...

## 4. 脚本文件和函数

函数：一个函数一个文件，文件名必须与函数名相同。

```matlab
function y = fun1(x)
y=x.^2+1;
end
```

匿名函数：`返回值 = @(形参列表) 表达式`

```matlab
clc,clear;
x=-5:0.1:5;
f=@(x) x.^2+1; %定义匿名函数
y=f(x); %调用匿名函数
plot(x,y)
figure %开辟新的绘图空间
fplot(f,[-5,5]) %使用匿名函数画图

figure
fplot(fun1,[-5,5])%使用文件函数画图
```

## 5. 数值积分

## 6. 线性方程组的解

求矩阵的秩`rank()`

矩阵的逆：`inv(矩阵)`，伪逆：`若a[3,4]，则a*pinv(a)是单位矩阵  `

行最简矩阵：`rref(矩阵)`结果比较异常。可以先把矩阵化成符号矩阵`sym(矩阵)`，再使用`rref()`

## 7. 枚举求解

```matlab
s=[];
for n = 63:63*2:N
   if rem(n,8)==1 && rem(n,5)==4 && rem(n,6)==3
       s=[s,n];
   end
end
s
```

# 二. 数据处理

## 1 数据组织

### 1.1 元胞数组

定义：

```matlab
a=cell(2,3)
b={1,rand(3);'dasffas',rand(2,1)}
```

索引：

```matlab
b{2,2} %查看元胞数组元素
b(:,1) %子元胞数组
```

元胞数组转矩阵：

```matlab
a={'张三','李四','王五';12,15,16};
cell2mat(a(2,:))
```

元胞拼接：

```matlab
a={'张三','李四','王五';12,15,16};
% []拼接
b=[a;{'L','F','F'}]
%结果
b =
  3×3 cell 数组
    {'张三'}    {'李四'}    {'王五'}
    {[  12]}    {[  15]}    {[  16]}
    {'L'   }    {'F'   }    {'F'   } 
% {}拼接
c={a;{'L','F','F'}}
c =
  2×1 cell 数组
    {2×3 cell}
    {1×3 cell}
```

例子：

![image-20220602220901677](https://typora-lff.oss-cn-guangzhou.aliyuncs.com/202206022209491.png)

```matlab
clc,clear;
%[num,txt,raw,custom] = xlsread(filename,sheet,xlRange,'',processFcn)
%a 读入的是数值矩阵，b 读入的字符串的元胞数组
[a,b]=xlsread('data1.xlsx');
a=[a(:,[1,2]);a(1:end-1,[4,5])]
b=[b(2:end,1);b(2:end-1,4)] %得到列向量
%b={b{2:end,1},b{2:end-1,4}} %或者 得到行向量
```

```matlab
%绘制图像
[a,b]=xlsread('data1.xlsx');
a=[a(:,[1,2]);a(1:end-1,[4,5])];
b={b{2:end,1},b{2:end-1,4}};
plot(a(:,1),a(:,2),'s');
text(a(:,1)+10,a(:,2),b);
```

### 1.2 结构体数组: 类似于数据库

定义：

```matlab
stu(1).name='LiMing'; stu(1).number='0101';
stu(1).sex='f'; stu(1).score=[90,80];
stu(2).name='LiHong'; stu(2).number='0102';
stu(2).sex='m'; stu(2).score=[88,80];
stu(2).name %查询
```

查询文件目录：

```matlab
f=dir('*.m')
```

```matlab
%例 2.7 读入附件 1 目录下所有的后缀名为 bmp 的图片文件，并把数据保存在元胞数组中。
clc, clear
f=dir('附件 1\*.bmp'); %读入所有 bmp 图像文件的信息，保存在结构数组 f 中
n=length(f) %计算 bmp 文件的个数
for i=1:n
 a{i}=imread(['附件 1\',f(i).name]); %把第 i 个图像数据保存到元胞数组
end
```

### 1.3 table数据

定义：每个变量必须具有相同的行数

```matlab
clc, clear
b=randi([0,6],10,4) %生成 10×4 的[0,6]上的随机整数矩阵
T2=array2table(b,'VariableNames',{'x1','x2','x3','y'})
summary(T2) %对表中数据进行统计
```

调用：

- 花括号{}的作用是{Rows,Columns}模式提取变量，形成数组。按照行列提取的数据要求是相互兼容的类型，不能一列是浮点数，另一列是字符串。
- 圆括号()的作用是(Rows,Columns)模式生成新的 table。
- 圆点.引用数据每次只能引用一列。

## 2 文本数据操作

### 2.1 txt文件

读取:

```matlab
a=textread('date1.txt')
b=load('date1.txt')
c=importdata('date1.txt') %万能函数
%结果
a =
     6     2     6     7     4     2     5     0
     4     9     5     3     8     5     8     0
     5     2     1     9     7     4     3     0
     7     6     7     3     9     2     7     0
b =
     6     2     6     7     4     2     5
     4     9     5     3     8     5     8
     5     2     1     9     7     4     3
     7     6     7     3     9     2     7
```

写入:（不能用）

```matlab
%矩阵数据
A = magic(5)
writematrix(A,'data2_20.csv')

%table数据
Pitch = [0.7;0.8;1;1.25;1.5];
Shape = {'Pan';'Round';'Button';'Pan';'Round'};
Price = [10.0;13.59;10.50;12.00;16.69];
Stock = [376;502;465;1091;562];
T = table(Pitch,Shape,Price,Stock)
writetable(T,'table2_14_1.csv')
rowNames = {'M4';'M5';'M6';'M8';'M10'};
T2 = table(Pitch,Shape,Price,Stock,'RowNames',rowNames)
writetable(T2,'table2_14_2.csv','WriteRowNames',true)

%元胞数据
C = {'Atkins',32,77.3,'M';'Cheng',30,99.8,'F';'Lam',31,80.2,'M'};
writecell(C,'data2_15.csv')
```

### 2.2 Excel文件

读取：

```matlab
a=importdata('data1.xlsx')
```

写入：

```matlab
clc, clear, rng(1) %进行一致性比较
A = randi([1,100],8,5) %生成 8×5 的随机整数矩阵
xlswrite('data2_16.xlsx',A)
```

### 2.3 mat格式

存储：

```matlab
save 文件名mydata 变量名 a b
```

读取：

```matlab
load mydata
```



## 3. 数据整理

### 3.1 缺失值的处理

```matlab
clc, clear
a = readtable('data2_23.xlsx')
id = {NaN,'',-19};
WrongPos = ismissing(a, id)
WrongData = a(any(WrongPos,2),:)
a = standardizeMissing(a,-19)
b = fillmissing(a,'previous') %使用缺失值前一行的值来填充缺失值。
c = rmmissing(a) %删除包含缺失值的记录
summary(c)
writetable(c, 'data2_24.xlsx') %保存到 Excel 文件供下面使用
```

### 3.2 排序

```matlab
clc, clear
a = readtable('data2_24.xlsx')
a.gender = categorical(a.gender) %gender 转换类型
summary(a)
b = sortrows(a,{'age'},{'descend'}) %age 数据降序排序
writetable(b,'data2_25.xlsx') %数据保存到 Excel 文件供下面使用
```

### 3.3 寻找异常值

```matlab
clc, clear
a = readtable('data2_25.xlsx')
b = isoutlier(a(:,[3:5]))
ind = find(any(b,2)) %找异常值所在的行数
```

### 3.4 合并数据源

```matlab
clc, clear
a = readtable('data2_25.xlsx')
b = readtable('data2_27.xlsx')
c = join(a,b,'Keys','gender') %合并两个数据源的数据
d = [c; c] %两份同样类型的样本数据合并
e = unique(d,'rows') %去掉重复数据
```

# 三. 绘制图像

## 3.1 散点图

二维散点图：

```matlab
clc, clear;
load seamount;
scatter(x,y,20,z,'fill')
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex','Rotation',0)
```

三维散点图：

```matlab
clc, clear;
load seamount;
scatter3(x,y,z,20,z,'fill')
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
zlabel('$z$','Interpreter','latex','Rotation',0)
```

多变量的散点图：

```matlab
clc, clear,close all;
load fisheriris;
tabulate(species) %频数表
gplotmatrix(meas,meas,species,'rgb','sdo',5) %绘制矩阵两两列之间的散点图
figure;
gscatter(meas(:,3),meas(:,4),species,'rgb','sod') %绘制分组的散点图
```

plot绘制散点图：

![image-20220603170734341](https://typora-lff.oss-cn-guangzhou.aliyuncs.com/202206031707729.png)

```matlab
clc, clear,close all;
load seamount;
subplot(121)
plot(x,y,'ro')
title('(A)二维散点图')
subplot(122)
plot3(x,y,z,'bo')
title('(B)三维散点图')
```

单位圆：

```matlab
clc, clear,close all;
t=0:0.01:2*pi;
x=cos(t);
y=sin(t);
plot(x,y,'r-'), axis square
figure
x=@(t)cos(t);y=@(t)sin(t);
fplot(x,y,[0,2*pi]); axis equal
```

圆锥曲线：

```matlab
clc, clear,close all;
subplot(121);
t=0:0.01:100;
x=t.*cos(t);y=t.*sin(t);
plot3(x,y,t);
subplot(122);
x=@(t)t.*cos(t);y=@(t)t.*sin(t);z=@(t)t;
fplot3(x,y,z,[0,100]) %函数绘图
```

圆锥表面：**表面图**

```matlab
clc, clear,close all;
subplot(121);
x=@(t,z)2*z.*cos(t);y=@(t,z)sqrt(2)*z.*sin(t);z=@(t,z)z;
fsurf(x,y,z,[0,2*pi,-5,5]); %表面图
subplot(122);
fimplicit3(@(x,y,z)x.^2/4+y.^2/2-z.^2,[-10,10,-10,10,-5,5])
```

莫比乌斯环：**网格图**

```matlab
clc, clear,close all;
x=@(s,t)(2+s/2.*cos(t/2)).*cos(t);
y=@(s,t)(2+s/2.*cos(t/2)).*sin(t);
z=@(s,t)s/2.*sin(t/2);
fmesh(x,y,z,[-1,1,0,2*pi])%网格图
```

等高线：

```matlab
a=contour(x,y,z);%绘制等高线
clabel(a);%标注高度
```

# 四. 线性规划

要素：

- 决策变量：未知数
- 约束条件：等式，不等式。
- 目标函数：最大值或者最小值

```matlab
clc, clear,close all;
prob=optimproblem('ObjectiveSense','max'); %目标函数最大化的优化问题
c=[4;3];b=[10;8;7];
a=[2 1 0 ;1 1 1];
x=optimvar('x',2,'LowerBound',0); %决策变量
prob.Objective=c'*x; %目标函数
prob.Constraints.con=a'*x<=b; %约束条件
[sol,fval,flag,out]=solve(prob);
sol.x
```

# 五. 整数规划

# 六. 非线性规划

只有凸规划可以求得全局最优解，其他情形只能求得局部最优解。

## 凸规划的标准型：

![image-20220603222428853](https://typora-lff.oss-cn-guangzhou.aliyuncs.com/202206032224322.png)

其中，**无约束优化问题的Matlab求解**：

```matlab
%[x,fval,exitflag]=fminunc(fun,x0) 其中，fun为目标函数，x0为初始值，exitflag>1是为局部最优解，为负数时结果不可靠
clc, clear,close all;
f=@(x) 100*(x(2)-x(1)^2)^2+(1-x(1))^2;
[s,f,flag]=fminunc(f,rand(1,2))
```

**有约束问题：**

标准型：

![image-20220603225528441](https://typora-lff.oss-cn-guangzhou.aliyuncs.com/202206032255763.png)

Matlab代码：

```matlab
%求解器格式[x,fval,flag]=fnimcon(fun,x0初始值,Aineq,bineq,Aeq,beq,lb,ub,nonlcon)
clc, clear,close all;
A=[-1 -2 0];b=-1;
lb=[0;-inf;-inf];ub=[inf,inf,inf];x0=rand(3,1);
f=@(x) -(2*x(1)+3*x(1).^2+3*x(2)+x(2).^2+x(3));
[x,fval,flag]=fmincon(f,x0,A,b,[],[],lb,[],@fun)

function [c,ceq]=fun(x)
c=[x(1)+2*x(1).^2+x(2)+2*x(2).^2+x(3)-10
    x(1)+x(1).^2+x(2)+x(2).^2-x(3)-50
    2*x(1)+x(1).^2+2*x(2)+x(3)-40];
ceq=x(1).^2+x(3)-2;
end
```

## 基于问题求解：

```matlab
%基于问题求解2018b版本不可用,2021a可以用
clc, clear,close all;
p=optimproblem('ObjectiveSense','max');
x=optimvar('x',3,1);
% fcn2optimexpr函数表达式转化成优化表达式
p.Objective=fcn2optimexpr(@(x)2*x(1)+3*x(1)^2+3*x(2)+x(2)^2+x(3),x);
x0.x=-100*rand(3,1);
p.Constraints.con1=[x(1)+2*x(1)^2+x(2)+2*x(2)^2+x(3)<=10
                    x(1)+x(1)^2+x(2)+x(2)^2-x(3)<=50
                    2*x(1)+x(1)^2+2*x(2)+x(3)<=40
                    1-x(1)-2*x(2)<=0
                    -x(1)<=0];
p.Constraints.con2=x(1)^2+x(3)==2;
opt=optimoptions('fmincon','Display','iter','Algorithm','active-set');
[s,f,flag,out]=solve(p,x0,'Options',opt)
```

技巧：目标函数不能直接使用绝对值函数abs，可以使用符号函数写abs表达式，再用`fcn2optimexpr`转化成目标函数

例题：

![image-20220604161216918](https://typora-lff.oss-cn-guangzhou.aliyuncs.com/202206041612478.png)

```matlab
clc, clear,close all;d0=readmatrix('data3_5');
prob=optimproblem;
x=optimvar('x',2,'LowerBound',0);
y=optimvar('y',2,'LowerBound',0);
z=optimvar('z',6,2,'LowerBound',0);
a=d0(1,:); b=d0(2,:); c=d0(3,:);
%目标函数
prob.Objective=fcn2optimexpr(@fun1,x,y,z,a,b);
prob.Constraints.con1=sum(z,2)==c';
prob.Constraints.con2=sum(z,1)<=20;
%设置初始值
x0.x=10*rand(2,1);
x0.y=10*rand(2,1);
x0.z=10*rand(6,2);
%设置求解器fmincon和算法sqp
opt=optimoptions('fmincon','Algorithm','sqp');
[sol,fval,flag,out]=solve(prob,x0,'Options',opt);
xx=sol.x, yy=sol.y, zz=sol.z

function obj=fun1(x,y,z,a,b)
obj=0;
for i=1:6
    for j=1:2
        obj=obj+z(i,j)*sqrt((x(j)-a(i))^2+(y(j)-b(i))^2)
    end
end
end
```

# 七. 图论

## 1. 构造

无向图：`graph=(vertex,edge,weight)`

有向图：`digraph=()`

使用**邻接矩阵**构造：

```matlab
clc, clear,close all;format shortG;
a=zeros(5);
a(1,2:4)=[5 3 7];a(2,[3,5])=[8,4];
a(3,[4,5])=[1 6];a(4,5)=2;
spa=sparse(a) %完全矩阵-->稀疏矩阵
ful=full(spa) %稀疏矩阵-->完全矩阵
G=graph(a,'upper') %无向赋权图
plot(G)

figure
b=zeros(4);
b(1,[2,3])=1;b(3,4)=1;b(4,1)=1;
D=digraph(b); %有向图
plot(D)
```

## 2. 最小生成树

### 2.1 Kruskal

原理：每次添加最小边，保证不构成圈，直到边v的数量等于顶点数量e-1

```matlab
clc, clear,close all;format shortG;
a=zeros(5);
a(1,2:4)=[5 3 7];a(2,[3,5])=[8,4];
a(3,[4,5])=[1 6];a(4,5)=2;
G=graph(a,'upper')
h=plot(G,'Layout','force','EdgeLabel',G.Edges.Weight)
%得到最小生成树,选择稀疏算法(即为Kruskal)
T=minspantree(G,'Method','sparse') 
T.Edges,T.Nodes  %查看最小生成树的结构
highlight(h,T)
```



### 2.2 Prim

原理：维护两个集合(顶点集合和边的集合)，从已有顶点向外延申边，选择最小边

```matlab
clc, clear,close all;format shortG;
a=zeros(5);
a(1,2:4)=[5 3 7];a(2,[3,5])=[8,4];
a(3,[4,5])=[1 6];a(4,5)=2;
G=graph(a,'upper')
h=plot(G,'Layout','force','EdgeLabel',G.Edges.Weight)
%得到最小生成树,默认稠密算法(即为Prim)
T=minspantree(G) 
T.Edges,T.Nodes  %查看最小生成树的结构
highlight(h,T)
```

例题：

![image-20220605101358644](https://typora-lff.oss-cn-guangzhou.aliyuncs.com/202206051014098.png)

```matlab
clc, clear,close all;format shortG;d0=importdata('data11_1.xlsx');
d0=d0.data(2:end,:);
d=dist(d0);  %求两两列之间的欧氏距离矩阵
G=graph(d)   %生成图
T=minspantree(G) 
L=sum(T.Edges.Weight) %求最小生成树的长度
plot(T,'EdgeLabel',T.Edges.Weight)
```

### 2.3 线性规划算法

```matlab
%待补充
```



## 3.最短路径算法

### 3.1 Dijkstra

条件：边的权值须要是非负数

解决的问题：从一点出发到其余各点的最短路径

原理：动态规划

![image-20220605120304791](https://typora-lff.oss-cn-guangzhou.aliyuncs.com/202206051203615.png)

```matlab
clc, clear,close all;format shortG;
%% 1.寻找所有可能存在状态，作为图的顶点
sm=[12 10 6 3];
s=[];
for x4=0:3
   for x3=0:6
       for x2=0:10
           t=[12-x2-x3-x4,x2,x3,x4];
           if t(1)>=0 & (~all(t) | ~all(sm-t))
               s=[s;t];
           end
       end
   end
end
s,n=size(s,1) %显示状态集合和状态个数
%% 2.寻找状态之间的邻接关系，构造邻接矩阵
w=zeros(n); %邻接矩阵初始化
for i=1:n
    for j=1:n
        vi=s(i,:);vj=s(j,:);
        ind=find(vi-vj); %查找改变分量的位置
        if length(ind)==2 & (~all(vj(ind)) | ~all(sm(ind)-vj(ind)))
           w(i,j)=1; 
        end
    end
end
%% 3.构造图，并求取最短路径
G=digraph(w); %构造有向赋权图
en=find(ismember(s,[4 4 4 0],'rows'));
[path,d]=shortestpath(G,1,en) %最短路径和距离
ps=s(path,:) %显示最短路径的遍历状态
```



### 3.2 Floyd

条件：允许边的权值是负数，但是要求每个圈的所有弧总和非负数

解决的问题：任意两点之间的最短路径

```matlab
clc, clear,close all;format shortG;n=5;
a=zeros(5);
a(1,2:4)=[5 3 7];a(2,[3,5])=[8,4];
a(3,[4,5])=[1 6];a(4,5)=2;
G=graph(a,'Upper');
plot(G,"EdgeLabel",G.Edges.Weight)
d1=distances(G) %最短路径矩阵，默认使用Dijkstra

%% 手写Floyd算法
b=a+a';b(b==0)=inf;
b([1:n+1:end])=0;
for k=1:n
    for i=1:n
        for j=1:n
            if b(i,j) > b(i,k)+b(k,j)
                b(i,j) = b(i,k)+b(k,j);
            end
        end
    end
end
b
```

### 3.3 整数规划求最短路径

```matlab
clc, clear,close all;format shortG;M=1e8;
a=zeros(6);
a(1,[2,5])=[18,15];a(2,[3:5])=[20,60,12];
a(3,[4,5])=[30,18];a(4,6)=10;a(5,6)=15;
b=a+a';b(b==0)=M;
p=optimproblem;
x=optimvar('x',6,6,'Type','integer','LowerBound',0,'UpperBound',1);
p.Objective=sum(sum(b.*x));
s=setdiff([1:6],[2,4]); %求集合的差集
p.Constraints=[sum(x(s,:),2)==sum(x(:,s))'
    sum(x(2,:))==1
    sum(x(:,2))==0
    sum(x(:,4))==1];
[sol,fval,flag,out]=solve(p)
xx=sol.x
[i,j]=find(xx);
ij=[i,j]
```

# 八. 微分方程模型

## 1 符号解：

求微分方程符号解的函数：`dsolve`

- 符号变量：
- 符号函数：

输出好看答案，用实时脚本。

例如：

```matlab
clc, clear,close all;format shortG;
syms x(t);
s=dsolve(diff(x)==x+sin(t),x(0)==1)
```

## 2. 数值解

`ode45`: `[t,y] = ode45(odefun,tspan,y0)`

```matlab
clc, clear,close all;format shortG;
% 符号解
syms y(x)
y=dsolve(diff(y)==-2*y+2*x^2+2*x,y(0)==1)
fplot(y,[0,0.5],1),hold on;
% 数值解
dy=@(x,y) -2*y+2*x^2+2*x;
[sx,sy]=ode45(dy,[0,0.5],1)
plot(sx,sy,'*');legend({"符号解","数值解"})
```

## 3. 人口预测模型

拟合参数：线性最小二乘法

```matlab
clc, clear,close all;format shortG;
x0=[1:10]';y0=randn(10,1);
xs=[x0,ones(10,1)]; %原表达式为：xs*ab=y0
ab=xs\y0
ab2=pinv(xs)*y0
xs2=[x0.*sin(x0),x0.^2]; %原表达式为：xs2*ab=y0
ab3=xs2\y0
ab4=pinv(xs2)*y0
```

```matlab
%简单的人口预测模型
clc, clear,close all;format shortG;
a=readmatrix(['E:\OneDrive\OneDrive - mail.sc' ...
    'ut.edu.cn\MathematicalMode' ...
    'ling\Book\数学建模算法与' ...
    '应用（第3版）程序及数据\06第6章  微分' ...
    '方程\data6_18.txt']);
y=a([2:2:6],:)';y=y(~isnan(y));
t=a([1:2:6],:)';t=t(~isnan(t));
plot(t,y);hold on; %绘制数据图

ts=t(2:end)-t(1);
ys=log(y(2:end))-log(y(1));
r=pinv(ts)*ys;
yt=@(t) y(1)*exp(r*(t-1790));
fplot(yt,[t(1),t(end)]) %绘制拟合图
```



# 九. 插值与拟合

## 1. 一维插值

插值函数：

```matlab
%  F = griddedInterpolant(X1,X2,...,Xn,V,Method)

```

三次样条插值推荐：

```matlab
%  pp = csape(x,[e1,y,e2],conds)
%  fnder(pp)	导数,pp结构
%  fnint(pp)	积分函数,pp结构
%  fnval(pp,x) 	对应点的值
clc, clear,close all;format shortG;
t0=0.15:0.01:0.18;v0=[3.5 1.5 2.5 2.8];
pp=csape(t0,v0)
x=0.15:0.00001:0.18;
plot(t0,v0,':or'),hold on;
plot(x,fnval(pp,x))
```

例题：求解国土面积和边界长度

```matlab
clc, clear,close all;format shortG;
a=readmatrix(['E:\OneDrive\OneDrive - mail.sc' ...
    'ut.edu.cn\MathematicalModeling\Book\数学' ...
    '建模算法与应用（第3版）程序及数据\05第5章  插' ...
    '值与拟合\data5_6.txt']);
x=a([1:3:end],:)';x=x(:);
y1=a([2:3:end],:)';y1=y1(:);
y2=a([3:3:end],:)';y2=y2(:);
f1=csape(x,y1);f2=csape(x,y2); %求样条结构数组
plot(x,fnval(f1,x)),hold on;
plot(x,fnval(f2,x))
S=integral(@(x)fnval(f2,x)-fnval(f1,x),min(x),max(x)) % 求国土面积
d1=fnder(f1);d2=fnder(f2);
L=integral(@(x) sqrt(1+fnval(d1,x).^2)+sqrt(1+fnval(d2,x).^2),min(x),max(x))
```

## 2. 二维插值

三次样条：`pp=csape({x0,y0},z0,conds,valconds)`

```matlab
clc, clear,close all;format shortG;
z0=readmatrix(['E:\OneDrive\OneDrive - mail.sc' ...
    'ut.edu.cn\MathematicalModeling\Book\数学' ...
    '建模算法与应用（第3版）程序及数据\05第5章  插' ...
    '值与拟合\data5_7.txt']);
x0=[0:100:1400];y0=[1200:-100:0];
pp=csape({x0,y0},z0');
x=0:10:1400;y=1200:-10:0;
z=fnval(pp,{x,y});z=z';
max_z=max(z);min_z=min(z);
subplot(121),h=contour(x,y,z);%画等高线
clabel(h); %标注等高线
subplot(122),mesh(x,y,z)
```

## 3. 拟合

### 3.1 线性拟合

手动计算参数，不推荐

```matlab
clc, clear,close all;format shortG;
t=[0:7]';y=[27 26.8 26.5 26.3 26.1 25.7 25.3 24.8]';
A=[t,ones(8,1)];
tb=mean(t);yb=mean(y); %求均值
ab=sum((t-tb).*(y-yb))/sum((t-tb).^2)
bb=yb-ab*tb
a_b=A\y % 使用 \ 拟合
cs2=polyfit(t,y,1) % 1维多项式polyfit拟合
cs3=fit(t,y,'poly1') %
cs4=fitlm(t,y) %
```

隐函数求参数：

```matlab
clc, clear,close all;format shortG;
x0=[5.764 6.286 6.759 7.168 7.408]';
y0=[0.648 1.202 1.823 2.526 3.36]';
A=[x0.^2,x0.*y0,y0.^2,x0,y0];
b=-ones(5,1);
c=A\b
fxy=@(x,y) c(1)*x.^2+c(2)*x.*y+c(3)*y.^2+c(4)*x+c(5)*y+1;
h=fimplicit(fxy,[3.5,8,-1,5]) %绘制隐函数
```

例子：拟合`z=a+blnx+cy`

```matlab
clc, clear,close all;format shortG;
x0=linspace(1,10,20);y0=linspace(3,20,20);
z0=1+2*log(x0)+3*y0;%产生虚拟数据
g=fittype('a+b*log(x)+c*y','dependent',{'z'},'independent',{'x','y'})%拟合模型
f=fit([x0',y0'],z0',g,'StartPoint',rand(1,3)) %利用模拟数据拟合
```

### 3.2 非线性拟合

`fit()`

```matlab
clc, clear,close all;format shortG;
x0=[1:50]';y0=2*cos(2*x0)+6*sin(2*x0);
lb=[-inf*ones(1,5),1];ub=[inf*ones(1,5),1];
[f,g]=fit(x0,y0,'fourier2','lower',lb,'upper',ub)
```

`lsqcurvefit()`

```matlab
clc, clear,close all;format shortG;
d=readmatrix(['E:\OneDrive\OneDrive - mail.scut.edu.c' ...
    'n\MathematicalModeling\Book\数学建模算法与应用（第' ...
    '3版）程序及数据\05第5章  插值与拟合\data5_18.txt']);
y=[d(:,2);d(1:end-1,7)];x=[d(:,3:5);d(1:end-1,8:10)];
% @(待拟合参数,自变量)
f=@(k,x)exp(-k(1)*x(:,1)).*sin(k(2)*x(:,2))+x(3).^2; 
lb=zeros(1,2);ub=2*ones(1,2);
% lsqcurvefit(f,参数初始值,自变量,因变量,[下界],[上界])
[cs,resmorm]=lsqcurvefit(f,rand(1,2),x,y,lb,ub)
```

图形界面拟合：

```matlab
clc, clear,close all;format shortG;
x=[0.81;0.91;0.13;0.91;0.63;0.098;0.28;0.55;0.96;0.96;0.16;0.97;0.96];size(x)
y=[0.17;0.12;0.16;0.0035;0.37;0.082;0.34;0.56;0.15;-0.046;0.17;-0.091;-0.071];size(y)
cftool %调用图形界面

%% 另一个函数.m文件
x=[0.81;0.91;0.13;0.91;0.63;0.098;0.28;0.55;0.96;0.96;0.16;0.97;0.96];size(x)
y=[0.17;0.12;0.16;0.0035;0.37;0.082;0.34;0.56;0.15;-0.046;0.17;-0.091;-0.071];size(y)
plot(F,x,y),hold on;
fplot(@(x)2*x-1,[0.4,1],'--');
fx=@(x)F(x)-(2*x-1) % 定义函数
x=fsolve(fx,rand) % 交点的横坐标
y=F(x) % 求交点的y坐标
```

# 十. 评价方法

## 评价之前的准备

- 数据有成本型和利润型，要进行一致化处理。
- 量级差异大，量纲不一致，需要进行无量纲处理。

```matlab
% 归一化处理
clc, clear,close all;format shortG;
d=readmatrix(['E:\OneDrive\OneDrive - mai'...
    'l.scut.edu.cn\MathematicalModelin'...
    'g\Book\数学建模算法与应用（第3版）程'...
    '序及数据\14第14章  综合评价与决策方法\data14_9_1.txt']);
b=sum(d,2)
[sb,index]=sort(b,'descend')
c1=zscore(d)  % 数据标准化
c2=(d-mean(d))./std(d)
check=c1-c2
```

## 主观分析法

1. 层次分析法

    ```matlab
    clc, clear,close all;format short G;%rat;
    a=readmatrix(['E:\OneDrive\OneDrive - mai'...
        'l.scut.edu.cn\MathematicalModelin'...
        'g\Book\数学建模算法与应用（第3版）程'...
        '序及数据\14第14章  综合评价与决策方法\data14_9_1.txt']);
    d=[1 2 4 4 4 6 6 6];
    d=fun1(d)
    [vec,val]=eigs(d,1) % 求最大特征值对应的特征向量
    w=vec/sum(vec) % 归一化权值
    e=a*w % 求加权结果
    [se,ind]=sort(e,'descend') % 对结果排序
    
    function nd=fun1(d)
    d=1./d;
    nd=[];
        for i=1:size(d,2)
            a=d(i)./d;
            nd=[nd;a];
        end
    end
    ```

2. 灰的关联分析法

    ```matlab
    
    ```

    

## 客观分析法
