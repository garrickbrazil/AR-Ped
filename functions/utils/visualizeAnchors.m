function gridim = visualizeAnchors(cls, blue)
    
    if ~exist('blue', 'var'), blue = false; end
    
    bsize = 2;
    bcolor = [0, 200, 200];
    
    %if size(cls,2)~=192
    %    cls = cls(:,:,end);
    %    cls = reshape(cls, [60, 45, 9]);
    %end
    
    a1 = repmat(imresize(cls(:,:,1)', 4), [1 1 3]);
    if(blue), a1(:,:,3) = 1 - a1(:,:,3); end
    
    a2 = repmat(imresize(cls(:,:,2)', 4), [1 1 3]);
    if(blue), a2(:,:,3) = 1 - a2(:,:,3); end
    
    a3 = repmat(imresize(cls(:,:,3)', 4), [1 1 3]);
    if(blue), a3(:,:,3) = 1 - a3(:,:,3); end
    
    a4 = repmat(imresize(cls(:,:,4)', 4), [1 1 3]);
    if(blue), a4(:,:,3) = 1 - a4(:,:,3); end
    
    a5 = repmat(imresize(cls(:,:,5)', 4), [1 1 3]);
    if(blue), a5(:,:,3) = 1 - a5(:,:,3); end
    
    a6 = repmat(imresize(cls(:,:,6)', 4), [1 1 3]);
    if(blue), a6(:,:,3) = 1 - a6(:,:,3); end
    
    a7 = repmat(imresize(cls(:,:,7)', 4), [1 1 3]);
    if(blue), a7(:,:,3) = 1 - a7(:,:,3); end
    
    a8 = repmat(imresize(cls(:,:,8)', 4), [1 1 3]);
    if(blue), a8(:,:,3) = 1 - a8(:,:,3); end
    
    a9 = repmat(imresize(cls(:,:,9)', 4), [1 1 3]);
    if(blue), a9(:,:,3) = 1 - a9(:,:,3); end
    
    a1 = padcolor(uint8(round(255*a1)), bsize, bcolor, 0, 0);
    a2 = padcolor(uint8(round(255*a2)), bsize, bcolor, 0, 0);
    a3 = padcolor(uint8(round(255*a3)), bsize, bcolor, 1, 0);
    a4 = padcolor(uint8(round(255*a4)), bsize, bcolor, 0, 0);
    a5 = padcolor(uint8(round(255*a5)), bsize, bcolor, 0, 0);
    a6 = padcolor(uint8(round(255*a6)), bsize, bcolor, 1, 0);
    a7 = padcolor(uint8(round(255*a7)), bsize, bcolor, 0, 1);
    a8 = padcolor(uint8(round(255*a8)), bsize, bcolor, 0, 1);
    a9 = padcolor(uint8(round(255*a9)), bsize, bcolor, 1, 1);
    
    gridim = [a1 a2 a3; a4 a5 a6; a7 a8 a9];
end

function im = padcolor(im, bsize, bcolor, lastcol, lastrow)

    for cind=1:3
        im(1:bsize, :,cind) = bcolor(cind);
        im(1:bsize, :,cind) = bcolor(cind);
        im(1:bsize, :,cind) = bcolor(cind);

        im(:, 1:bsize,cind) = bcolor(cind);
        im(:, 1:bsize,cind) = bcolor(cind);
        im(:, 1:bsize,cind) = bcolor(cind);

        if lastcol
            im(:, end+1-bsize:end,cind) = bcolor(cind);
            im(:, end+1-bsize:end,cind) = bcolor(cind);
            im(:, end+1-bsize:end,cind) = bcolor(cind);
        end
        
        if lastrow
            im(end+1-bsize:end,:, cind) = bcolor(cind);
            im(end+1-bsize:end,:, cind) = bcolor(cind);
            im(end+1-bsize:end,:, cind) = bcolor(cind);
        end
    end

end
