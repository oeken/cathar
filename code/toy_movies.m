clear;
clc;

%read the excel file
filename='toy_movies.xlsx';
[num,txt,raw]=xlsread(filename);

R = num';
users = txt(2,3:22);
movies = txt(3:22,2);
