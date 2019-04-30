function parent_node = get_parent(x_near, nodes, edges)
% Given node v, this function finds the parent node u of node v such that 
% (u,v) belongs to the edge set.

    parent_node = TreeNode;
    edge_from   = TreeNode;
    edge_to     = TreeNode;
    if isequaln(x_near,nodes(1)) % rootnode has no parent - It is its own parent
        parent_node = x_near;        
    else
        for i = 1:length(edges(:,1))                     
            edge_from = edges(i,1);
            edge_to   = edges(i,2);               
            if isequaln(x_near, edge_to) 
                parent_node = edge_from; 
                break;
            end
        end   
    end 
end

