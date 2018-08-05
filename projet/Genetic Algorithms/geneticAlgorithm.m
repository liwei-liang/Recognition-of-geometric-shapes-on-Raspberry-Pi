Be=[ 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1  1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1  1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1  1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1];
A =rand(100,64);
B=floor(2*A);
Fdis=zeros(100,1);
Rpool=[];
Cpool=[];
used = 0;
fois = 0;
figure
hold on
while(fois < 150)
Fdis=zeros(100,1);
Rpool=[];
Cpool=[];
used = 0;

for i = 1: 1 :100
    for j = 1: 1: 64
        Fdis(i,1) = Fdis(i,1) + abs(B(i,j)-Be(1,j));
    end
end
for i = 1: 1: 100
    if(Fdis(i,1)<25)
        for k = 1 : 1 :3
          Rpool(used+k,:)=B(i,:);
        end
        used = used + 3;
    end
    if(24<Fdis(i,1)&&Fdis(i,1)<48)
         for k = 1 : 1 :2
          Rpool(used+k,:)=B(i,:);
         end
        used = used + 2;
    end
    if(Fdis(i,1)>48)
          Rpool(used+1,:)=B(i,:);
          used = used + 1;
    end
end
choose = rand(90,1);
line = floor(size(Rpool,1) * choose)+1;%随机选择Rp向Cp投入的行
[C,I]= sort(Fdis);
for p = 1 : 1 :10
    Cpool(p,:) = B(I(p,1),:);
end

for m = 11 : 1 : 100
    Cpool(m,:) = Rpool(line(m-10,1),:);
end
Cspool = Cpool;
for o = 1 : 1 : 45
    pointToChange = ceil(rand*64);
    Cpool(o+10,pointToChange:end)= Cspool(o+55,pointToChange:end);
    Cpool(o+55,pointToChange:end)= Cspool(o+10,pointToChange:end);
end
pointChange = rand(6,2);
pointChange(:,1)=floor(pointChange(:,1)*90)+11;
pointChange(:,2)=floor(pointChange(:,2)*64)+1;
for i = 1 : 1 : 6
    Cpool(pointChange(i,1),pointChange(i,2)) = xor( Cpool(pointChange(i,1),pointChange(i,2)),1);
end
B = Cpool;
fois = fois +1;
E=reshape(B(1,:),8,8);
imagesc(E);
drawnow;
end

    
    
