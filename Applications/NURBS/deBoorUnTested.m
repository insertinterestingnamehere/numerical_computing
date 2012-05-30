t=linspace(0,1,100);
U=[ 0 0 0 0 1 1 1 1];
Q3D=bspline(t,P,U); 
% % % --------------------------------
% % % Author: begtostudy
% % % MSN:begtostudy@hotmail.com
% % % Email : begtostudy@gmail.com
% % % --------------------------------
function Q = bspline(t,P,U ,w)
% A Single B-Spline/NURBS for given control points.
%
%Example:
% P=[292 280 321 356;
% 196 153 140 148;
% -56 75 140 248];
% %pols num =4
% 
% t=linspace(0,1,100);
% Q3D=bspline(t,P);
% 
% figure
% plot3(Q3D(1,:),Q3D(2,:),Q3D(3,:),'b','LineWidth',2),
% hold on
% plot3(P(1,:),P(2,:),P(3,:),'g:','LineWidth',2)        % plot control polygon
% plot3(P(1,:),P(2,:),P(3,:),'ro','LineWidth',2)     % plot control points
% view(3);
% box;
    if (nargin<3)
      U=linspace(0,1,size(P,2)+size(P,2));
    end
e0=find(t>=U(size(P,2)),1,'first');
e1=find(t<=U(end-size(P,2)+1),1,'last');
t=t(e0:e1);
Q=zeros(3,length(t));
     if (nargin<4)
       w=ones(1,size(P,2));
     end
    for k=1:length(t)
        rational=0;
        for j=1:size(P,2)
            tmp=deboor(U,size(P,2)-1,j,t(k));
            if (nargin==4)
                rational =rational+w(j)*tmp;
            end
            Q(:,k)=Q(:,k)+w(j)*P(:,j)*tmp;
        end
        if (nargin==4)
            Q(:,k)=Q(:,k)/rational;
        end
    end
end
 
function N=mydeboor(U,k,i,u)
i=i+1;
N=deboor(U,k,i,u);
end
 
function N=deboor(U,k,i,u)
    if isequal(k,0)  
        %the u may equal to last U
        if u>=U(i) && abs(u-U(i+1))<1e-6 && i+1==(length(U)/2+1)
            N=1;
        elseif u>=U(i) && u<U(i+1)
           N=1;
        else
            N=0;
        end
    else
        N=division((u-U(i))*deboor(U,k-1,i,u),U(i+k)-U(i))...
        +division((U(i+k+1)-u)*deboor(U,k-1,i+1,u),U(i+k+1)-U(i+1));
    end
end
 
function d=division(a,b)
    if (abs(a)<1e-6) && (abs(b)<1e-6) 
        d=0;
    else
        d=a/b;
    end
end
