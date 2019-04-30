function cost = compute_cost(node, nodes, edges)
% Cost(v) = Cost(Parent(v)) + c(Line(Parent(v), v))

    cost = 0;    
    while ~cost    
        parent_node = get_parent(node, nodes, edges);
        if isequaln(parent_node,node)
            break;
        else
            cost = cost + compute_cost(parent_node, nodes, edges);  
            cost = cost + compute_distance([parent_node.x parent_node.y], [node.x, node.y]);
        end                
        node = parent_node;
    end    
    

end

