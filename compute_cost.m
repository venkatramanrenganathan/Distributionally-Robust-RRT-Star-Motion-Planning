function cost = compute_cost(node, T)
% Cost(v) = Cost(Parent(v)) + c(Line(Parent(v), v)) - Implementation

    cost = 0;    
    while ~cost    
        parent_node = get_parent(node, T);
        if isequaln(parent_node,node)
            break;
        else
            cost = cost + compute_cost(parent_node, T);  
            cost = cost + (parent_node - node)*P0*(parent_node - node)';
        end                
        node = parent_node;
    end        

end

