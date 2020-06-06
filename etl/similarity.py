
import numpy as np

def similarity(data):
    
    text_clusters = data['text_clusters']
    numeric_clusters = data['numeric_clusters']
    
    text_names = data['text_names']
    numeric_names = data['numeric_names']
    
    edges = []
    for t_c, t_n in zip(text_clusters, numeric_clusters):
        for i in range(len(t_c)):
            for j in range(i, len(t)):
                A_elem, A_name = t_c[i], t_n[i]
                B_elem, B_name = t_c[j], t_n[j]
                A_info = split(A_name, '.')
                B_info = split(B_name,'.')
                if not np.equal(A_info[:-1], B_info[:-1]):
                    edges.append(np.concat(A_info, B_info))
    return rows
                
        