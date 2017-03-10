X=load('a.txt');
X=(reshape(X(1:10000),100,100))
imagesc(X),colorbar,colormap gray;
