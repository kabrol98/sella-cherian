
import numpy as np 

def similarity(data):
    
    text_clusters = data['text_clusters']
    numeric_clusters = data['numeric_clusters']
    
    text_names = data['text_names']
    numeric_names = data['numeric_names']
    sheet_record = {}
    edges = []
    sheet_edges = []
    for t_c, t_n in zip(text_clusters, text_names):
        for i in range(len(t_c)):
            for j in range(i, len(t_c)):
                print(t_c, t_n)
                A_elem, A_name = t_c[i], t_n[i]
                B_elem, B_name = t_c[j], t_n[j]
                A_info = A_name.split('.')
                B_info = B_name.split('.')
                print(A_info, B_info)
                for a,b in zip(A_info[:-1],B_info[:-1]):
                    if not a == b:
                        edges.append(A_info + B_info)
                        srow = A_info[:-1]+ B_info[:-1]
                        if sheet_record[tuple(srow)]:
                            sheet_edges.append(srow)
                            sheet_record[tuple(srow)] = True
                        
                        break
    
    for t_c, t_n in zip(numeric_clusters, numeric_names):
        for i in range(len(t_c)):
            for j in range(i, len(t_c)):
                print(t_c, t_n)
                A_elem, A_name = t_c[i], t_n[i]
                B_elem, B_name = t_c[j], t_n[j]
                A_info = A_name.split('.')
                B_info = B_name.split('.')
                print(A_info, B_info)
                for a,b in zip(A_info[:-1],B_info[:-1]):
                    if not a == b:
                        edges.append(A_info + B_info)
                        srow = A_info[:-1]+ B_info[:-1]
                        if sheet_record[tuple(srow)]:
                            sheet_edges.append(srow)
                            sheet_record[tuple(srow)] = True
                        break
    return {'columns': edges, 'sheets': sheet_edges}
                
        