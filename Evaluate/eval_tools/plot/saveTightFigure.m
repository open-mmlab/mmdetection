function saveTightFigure(h,outfilename)
% SAVETIGHTFIGURE(OUTFILENAME) Saves the current figure without the white
%   space/margin around it to the file OUTFILENAME. Output file type is
%   determined by the extension of OUTFILENAME. All formats that are
%   supported by MATLAB's "saveas" are supported. 
%
%   SAVETIGHTFIGURE(H, OUTFILENAME) Saves the figure with handle H. 
%
% E Akbas (c) Aug 2010
% * Updated to handle subplots and multiple axes. March 2014. 
%

if nargin==1
    hfig = gcf;
    outfilename = h;
else 
    hfig = h;
end

%% find all the axes in the figure
hax = findall(hfig, 'type', 'axes');

%% compute the tighest box that includes all axes
tighest_box = [Inf Inf -Inf -Inf]; % left bottom right top
for i=1:length(hax)
    set(hax(i), 'units', 'centimeters');
    
    p = get(hax(i), 'position');
    ti = get(hax(i), 'tightinset');
    
    % get position as left, bottom, right, top
    p = [p(1) p(2) p(1)+p(3) p(2)+p(4)] + ti.*[-1 -1 1 1];
    
    tighest_box(1) = min(tighest_box(1), p(1));
    tighest_box(2) = min(tighest_box(2), p(2));
    tighest_box(3) = max(tighest_box(3), p(3));
    tighest_box(4) = max(tighest_box(4), p(4));
end

%% move all axes to left-bottom
for i=1:length(hax)
    if strcmp(get(hax(i),'tag'),'legend')
        continue
    end
    p = get(hax(i), 'position');
    set(hax(i), 'position', [p(1)-tighest_box(1) p(2)-tighest_box(2) p(3) p(4)]);
end

%% resize figure to fit tightly
set(hfig, 'units', 'centimeters');
p = get(hfig, 'position');

width = tighest_box(3)-tighest_box(1);
height =  tighest_box(4)-tighest_box(2); 
set(hfig, 'position', [p(1) p(2) p(3) p(4)]);

%% set papersize
set(hfig,'PaperUnits','centimeters');
set(hfig,'PaperSize', [width height]);
set(hfig,'PaperPositionMode', 'manual');
set(hfig,'PaperPosition',[0 0 width height]);


%% save
saveas(hfig,outfilename);
