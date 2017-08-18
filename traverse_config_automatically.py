import config



def update_config(value_combination, config_to_traverse):
    for config_idx in range(len(config_to_traverse)):
        config_str = config_to_traverse[config_idx]['config_to_assign']
        value_idx = value_combination[config_idx]
        value = config_to_traverse[config_idx]['values_to_try'][value_idx]
        exec(config_str+' = '+str(value))

def get_config_generator(config_to_traverse):
    max_depth = len(config_to_traverse)

    traverse_stack = [0]
    depth = 0
    while depth >= 0:
        if depth == max_depth:
            update_config(traverse_stack, config_to_traverse)
            del traverse_stack[-1]
            depth -= 1
            traverse_stack[depth] += 1
            yield
        else:
            assert len(traverse_stack)-1 == depth
            now_value_idx = traverse_stack[depth]
            max_value_idx = len(config_to_traverse[depth]['values_to_try'])
            if now_value_idx < max_value_idx:
                traverse_stack.append(0)
                depth += 1
            else:
                del traverse_stack[-1]
                depth -= 1
                if depth >= 0:
                    traverse_stack[depth] += 1

