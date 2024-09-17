def pre_walk_tree_sitter_with_hyper(node, index, edge_index, tokenizer):
    _, edge_index, tokens, ast_hyperedge_indexs, line_hyperedge_indexs, word_hyperedge_indexs = pre_walk_tree_sitter_with_multi_hyper(
        node, index, edge_index, tokenizer)

    hyperedge_indexs = [[], []]
    edge_types = []

    line_dict = {}
    for line_hyperedge_index in line_hyperedge_indexs:
        if line_hyperedge_index[1] not in line_dict:
            line_dict[line_hyperedge_index[1]] = [line_hyperedge_index[0]]
        else:
            line_dict[line_hyperedge_index[1]].append(line_hyperedge_index[0])

    for word_hyperedge_index in word_hyperedge_indexs:
        hyperedge_indexs[0].append(word_hyperedge_index[0])
        hyperedge_indexs[1].append(word_hyperedge_index[1])
        edge_types.append(0)
    for ast_hyperedge_index in ast_hyperedge_indexs:
        hyperedge_indexs[0].append(ast_hyperedge_index[0])
        hyperedge_indexs[1].append(ast_hyperedge_index[1])
        edge_types.append(1)
    for edge in line_dict:
        if len(line_dict[edge]) > 1:
            for node in line_dict[edge]:
                hyperedge_indexs[0].append(node)
                hyperedge_indexs[1].append(edge_index)
                edge_types.append(2)
            edge_index += 1

    return tokens, hyperedge_indexs, edge_types


def pre_walk_tree_sitter_with_multi_hyper(node, index, edge_index, tokenizer):
    tokens = []
    ast_hyperedge_indexs = []
    line_hyperedge_indexs = []
    word_hyperedge_indexs = []

    if not node.children:
        sub_tokens = tokenizer.tokenize(node.text.decode('UTF-8'))

        if len(sub_tokens) > 1:
            for i in range(len(sub_tokens)):
                tokens.append(sub_tokens[i])
                word_hyperedge_indexs.append([index, edge_index])
                if node.start_point[0] == node.end_point[0]:
                    line_hyperedge_indexs.append([index, node.start_point[0]])
                index += 1
            edge_index += 1
        elif len(sub_tokens) == 1:
            tokens.append(sub_tokens[0])
            if node.start_point[0] == node.end_point[0]:
                line_hyperedge_indexs.append([index, node.start_point[0]])
            index += 1

        return index, edge_index, tokens, ast_hyperedge_indexs, line_hyperedge_indexs, word_hyperedge_indexs

    num_leaf_child = 0
    leaf_child_indexs = []
    for child in node.children:
        index, edge_index, child_tokens, child_ast_hyperedge_indexs, child_line_hyperedge_indexs, child_word_hyperedge_indexs = pre_walk_tree_sitter_with_multi_hyper(
            child, index, edge_index, tokenizer)
        tokens.extend(child_tokens)
        ast_hyperedge_indexs.extend(child_ast_hyperedge_indexs)
        line_hyperedge_indexs.extend(child_line_hyperedge_indexs)
        word_hyperedge_indexs.extend(child_word_hyperedge_indexs)

        if not child.children:
            num_leaf_child += 1
            leaf_child_indexs.extend([index - i for i in range(len(child_tokens), 0, -1)])

    if num_leaf_child > 1:
        for leaf_child_index in leaf_child_indexs:
            ast_hyperedge_indexs.append([leaf_child_index, edge_index])
        edge_index += 1

    return index, edge_index, tokens, ast_hyperedge_indexs, line_hyperedge_indexs, word_hyperedge_indexs
