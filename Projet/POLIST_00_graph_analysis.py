import pandas as pd
import numpy as np
# import geopandas as gpd
import matplotlib.pyplot as plt
import re


def get_cols(datas):
    # Récupération des états
    state_cols = []
    cat_cols = []
    type_cols = []
    for col in datas.columns:
        searched_states = re.findall(r"^customer_state__.*", col)
        searched_cat = re.findall(r"^category__.*", col)
        searched_type = re.findall(r"^type_.*", col)
        if len(searched_states) > 0:
            state_cols.append(searched_states[0])
        if len(searched_cat) > 0:
            cat_cols.append(searched_cat[0])
        if len(searched_type) > 0:
            type_cols.append(searched_type[0])
    return state_cols, cat_cols, type_cols


def get_categories(cat_sums, total_cat_sum, ord_bool):
    cat_sums = cat_sums.sort_values(by= "values", ascending= ord_bool).reset_index(drop= True).to_numpy()
    
    list_val = []
    for cat_sum in cat_sums[:3, :]:
        col = " ".join(re.findall(r"^category__(.*)_sum", cat_sum[0])[0].split("_"))
        list_val.append(col)
        list_val.append(np.round(cat_sum[1] / total_cat_sum, 3) *100)
    return list_val



def segmentation(datas, conditions, segments_names):
    """datas: datas to apply segmentation on;
    condtions: list of conditions to get cluster (ex: [datas["labels"] == 0), ...];
    segment_names => list of names of segments or clusters"""

    final_cols = ["segment_name", "nb_clients", "perc_clients", "payment_mean_value", # 0-3
        "montant_CA", "perc_CA", "buyed_prod_mean", "prod_price_mean", "buyed_prod_nb", # 4-8
        "perc_buyed_prod_nb", "perc_boleto_payment", "perc_credit_card_payment", # 9-11
        "perc_debit_card_payment", "perc_voucher_payment", "best_cat_1", "perc_best_cat_1", # 12-15
        "best_cat_2", "perc_best_cat_2", "best_cat_3", "perc_best_cat_3", "worst_cat_1", # 16-20
        "perc_worst_cat_1", "worst_cat_2", "perc_worst_cat_2", "worst_cat_3", "perc_worst_cat_3", # 21-25
        "mean_last_purchase_day", "mean_frequency", "mean_scores", "best_cust_state", # 26-29
        "relat_perc_best_cust_state", "abs_perc_best_cust_state", "worst_cust_state", # 30-32
        "relat_perc_worst_cust_state", "abs_perc_worst_cust_state"] # 33-34

    # Création des feature pour le jeu final
    final_datas = {}
    for col in final_cols:
        final_datas[col] = []

    # Tri des features
    state_cols, cat_cols, type_cols = get_cols(datas)

    ### Valeurs extraites du jeu de données initial
    ca_total = datas["pay_sum_sum"].sum()
    nb_prod_total = datas["nb_product_sum_sum"].sum()
    total_customers = len(datas)

    for condition, segment_name in zip(conditions, segments_names):
        # J'extrais les clients du segment concerné
        segment_datas = datas[condition]

        final_datas[final_cols[0]].append(segment_name)
        final_datas[final_cols[1]].append(len(segment_datas))
        perc_clients = np.round(len(segment_datas) / total_customers, 3) *100
        final_datas[final_cols[2]].append(perc_clients)
        # Je calcule les valeurs à mettre en avant 
        # que je stocke dans une liste
        pay_mean_val = np.round(segment_datas["pay_sum_sum"].mean(), 2)
        final_datas[final_cols[3]].append(pay_mean_val)
        montant_ca = np.round(segment_datas["pay_sum_sum"].sum(), 2)
        final_datas[final_cols[4]].append(montant_ca)

        perc_ca = np.round(montant_ca / ca_total, 3) *100
        final_datas[final_cols[5]].append(perc_ca)
        
        buyed_prod_mean = np.round(segment_datas["nb_product_sum_sum"].mean(), 2)
        final_datas[final_cols[6]].append(buyed_prod_mean)

        prod_price_mean = np.round(pay_mean_val / buyed_prod_mean, 2)
        final_datas[final_cols[7]].append(prod_price_mean)

        buyed_prod_nb = segment_datas["nb_product_sum_sum"].sum()
        final_datas[final_cols[8]].append(buyed_prod_nb)

        perc_buyed_prod = np.round(buyed_prod_nb / nb_prod_total, 3) *100
        final_datas[final_cols[9]].append(perc_buyed_prod)

        payments = segment_datas[type_cols].sum()
        total_payments = payments.sum()
        for payment, feat in zip(payments, final_cols[10:14]):
            final_datas[feat].append(np.round(payment / total_payments, 3) *100)

        cat_sums = segment_datas[cat_cols].sum().reset_index().rename(columns= 
            {"index": "categories", 0: "values"})
        total_cat_sum = cat_sums["values"].sum()
        # Je garde les 3 meilleures catégories puis les 3 pires
        cat_values_max = get_categories(cat_sums, total_cat_sum, False)
        cat_values_min = get_categories(cat_sums, total_cat_sum, True)
        for feat, val in zip(final_cols[14: 26], cat_values_max + cat_values_min):
            final_datas[feat].append(val)

        mean_last_purchase = np.round(segment_datas["days_last_purchase_min"].mean(), 1)
        final_datas[final_cols[26]].append(mean_last_purchase)

        mean_freq = np.round(segment_datas["frequency"].mean(), 1)
        final_datas[final_cols[27]].append(mean_freq)

        mean_scores = np.round(segment_datas["init_score_mean"].mean(), 1)
        final_datas[final_cols[28]].append(mean_scores)

        cust_sums = segment_datas[state_cols].sum().reset_index().rename(columns= 
            {"index": "state", 0: "values"}).sort_values(by= "values", ascending= False)
        total_segm_cust = cust_sums["values"].sum()
        cust_max = cust_sums.iloc[:1, :].to_numpy()[0]
        cust_min = cust_sums.iloc[-1:, :].to_numpy()[0]

        final_datas[final_cols[29]].append(re.findall(r"^customer_state___(.*)", cust_max[0])[0])
        final_datas[final_cols[30]].append(np.round(cust_max[1] / total_segm_cust, 3) *100)
        final_datas[final_cols[31]].append(np.round(cust_max[1] / total_customers, 3) *100)
        final_datas[final_cols[32]].append(re.findall(r"^customer_state___(.*)", cust_min[0])[0])
        final_datas[final_cols[33]].append(np.round(cust_min[1] / total_segm_cust, 3) *100)
        final_datas[final_cols[34]].append(np.round(cust_min[1] / total_customers, 3) *100)


    return pd.DataFrame(final_datas)


# def get_states(datas):
#     # Récupération des états
#     state_cols = []
#     cat_cols = []
#     for col in datas.columns:
#         searched_states = re.findall(r"^customer_state__.*", col)
#         searched_cat = re.findall(r"^category__.*", col)
#         if len(searched_states) > 0:
#             state_cols.append(searched_states[0])
#         if len(searched_cat) > 0:
#             cat_cols.append(searched_cat[0])
#     return state_cols, cat_cols


# def sum_mean(cats_cols, states_cols, sum_col, mean_col, graph_name, segment):
#     # boucle sur les colonnes "customer_state__*"
#     states_datas = {}
#     cat_datas = {}
#     bool_1 = True
#     quantiles = pd.DataFrame([], index= [0, 1, 2])
#     for sum_c, mean_c, name in zip(sum_col, mean_col, graph_name):
#         if states_datas.get(name + "_sum") is None:
#             states_datas[name + "_sum"] = []
#         if states_datas.get(name + "_mean") is None:
#             states_datas[name + "_mean"] = []
#         if states_datas.get(name + "_quant") is None:
#             states_datas[name + "_quant"] = []
#         # print(sum_c, " - ", mean_c, " - ", name)

#         for col in states_cols:
#             if bool_1:
#                 searched_state = re.findall(r"^customer_state___(.*)", col)[0]
#                 if states_datas.get("state") is None:
#                     states_datas["state"] = []
                
#                 states_datas["state"].append(searched_state)
#                 states_datas[name + "_sum"].append(segment[col].sum())
        
        
#             if sum_c is not None:
#                 # somme sur sum_col par état
#                 somme = segment[segment[col] > 0][sum_c].sum()
#                 states_datas[name + "_sum"].append(somme)
#                 # moyenne sur mean_col par état
#                 moy = np.round(segment[segment[col] > 0][mean_c].mean(), 2)
#                 if np.isnan(moy):
#                     moy = 0
#                 states_datas[name + "_mean"].append(moy)
                
#             else:
#                 # moyenne sur mean_col par état
#                 moy = np.round(segment[segment[col] > 0][mean_c].mean(), 2)

#                 if np.isnan(moy):
#                     moy = -1
#                 states_datas[name + "_mean"].append(moy)

#         if bool_1:
#             bool_1 = False

#             for col in cats_cols:
#                 searched_cat = re.findall(r"^category__(.*)", col)[0]
#                 if cat_datas.get("categories") is None:
#                     cat_datas["categories"] = []
#                 if cat_datas.get("category_val") is None:
#                     cat_datas["category_val"] = []

#                 cat_datas["categories"].append(searched_cat)

#                 somme_cat = segment[col].sum()
                
#                 cat_datas["category_val"].append(somme_cat)

#         quant_col = name + "_mean"
#         # quantile sur mean pour la taille des points
#         tmp_df = pd.DataFrame(states_datas[quant_col])
#         quantiles[quant_col] = tmp_df[tmp_df[:] >= 0].quantile([0.25, 0.5, 0.75]).round(3).reset_index(drop= True)
        
#         df = pd.DataFrame(states_datas[quant_col], columns= [quant_col])
#         for i in reversed(range(len(quantiles) +2)):
#             if df[quant_col].min() < 0:
#                 if i == 0:
#                     df.loc[df[quant_col] < 0, "quantiles"] = i +1
#                 else:
#                     df.loc[df[quant_col] <= quantiles.iloc[i -3, -1], "quantiles"] = i +1
#                     if i == len(quantiles) +1:
#                         df.loc[df[quant_col] > quantiles.iloc[i -3, -1], "quantiles"] = i +1

#             else:
#                 i -= 2
#                 if i >= 0:
#                     df.loc[df[quant_col] <= quantiles.iloc[i, -1], "quantiles"] = i +1
#                     if i == len(quantiles) -1:
#                         df.loc[df[quant_col] > quantiles.iloc[i, -1], "quantiles"] = i +2

#         states_datas[name + "_quant"] = df.loc[:, "quantiles"].tolist()
        
#     return pd.DataFrame(states_datas), quantiles, pd.DataFrame(cat_datas)


# def get_label(transf_quant, quantiles_datas, quant_col, orig_quant_datas):
    
#     transf_quant_name = transf_quant.name
#     transf_quant = pd.DataFrame(transf_quant, columns= [transf_quant.name], index= transf_quant.index)
#     nb_quant = np.sort(transf_quant[transf_quant_name].unique())
    
#     # print(nb_quant)
#     for num, i in enumerate(nb_quant):
#         if orig_quant_datas.min() < 0:
#             if num == 0:
#                 transf_quant.loc[transf_quant[transf_quant_name] == i, "labels"] = "n/a"
#             elif num == 1:
#                 transf_quant.loc[transf_quant[transf_quant_name] == i, "labels"] = str(orig_quant_datas[
#                     orig_quant_datas >= 0].min()) + " - " + str(quantiles_datas.loc[0, quant_col])
#             elif num == len(nb_quant) -1:
#                 transf_quant.loc[transf_quant[transf_quant_name] == i, "labels"] = str(quantiles_datas.loc[
#                     len(quantiles_datas) -1, quant_col]) + " - " + str(orig_quant_datas.max())
#             else:
#                 transf_quant.loc[transf_quant[transf_quant_name] == i, "labels"] = \
#                     str(quantiles_datas.loc[num -2, quant_col]) + " - " + str(quantiles_datas.loc[num -1, quant_col])
        
#         else:
#             if num == 0:
#                 transf_quant.loc[transf_quant[transf_quant_name] == i, "labels"] = str(orig_quant_datas.min()) + \
#                     " - " + str(quantiles_datas.loc[num, quant_col])
#             elif num == len(nb_quant) -1:
#                 transf_quant.loc[transf_quant[transf_quant_name] == i, "labels"] = str(quantiles_datas.loc[
#                     len(quantiles_datas) -1, quant_col]) + " - " + str(orig_quant_datas.max())
#             else:
#                 transf_quant.loc[transf_quant[transf_quant_name] == i, "labels"] = \
#                     str(quantiles_datas.loc[num -1, quant_col]) + " - " + str(quantiles_datas.loc[num, quant_col])

#     transf_quant.sort_values(by= transf_quant_name, inplace= True)

#     return transf_quant[~transf_quant.duplicated(subset= "labels", keep= "first")]["labels"].tolist()


# def to_log(X):
#     return np.log(X +1)


# def cat_graph_transf(cats_datas, perc_threshold):
#     # Dans "cats_datas" se trouvent deux features: "categories" et "category_val"
#     # Pour chaque "categories", j' extrais les pourcentages de 
#     # "category_val"/"category_val".sum() et j'ajoute les autres à une liste dont
#     # je ferai la somme à la fin
    
#     colors = {0: "blue", 1: "orange", 2: "green", 3: "red", 4: "purple", 5: "brown", 6: "pink", 
#         7: "olive", 8: "cyan"}
#     wanted_list = []
#     others_list = []
#     for idx, row in cats_datas.iterrows():
#         color = colors[idx % 9]
#         if row["category_val"] / cats_datas["category_val"].sum() >= perc_threshold:
#             label = " ".join(row["categories"].split("_")[:-1])
#             wanted_list.append([label, row["category_val"], 0.1, color])
#         else:
#             others_list.append(row["category_val"])
#     others_sum = np.sum(others_list)
#     wanted_list.append(["Autres", others_sum, 0, "grey"])
#     return pd.DataFrame(wanted_list, columns= ["labels", "values", "explode", "colors"]).sort_values(by= 
#         "values", ascending= False)


# def print_graph(map_datas, cats_datas, sum_mean_datas, quantiles, graph_name):
#     # Pie chart des catégories
#     # cats_datas = cats_datas.sort_values(by= cats_datas.columns[1])

#     cat_graph_datas = cat_graph_transf(cats_datas, 0.025)

#     fig, ax = plt.subplots(figsize=(10, 10))

#     # explode = [0.1 if x / cats_datas.iloc[:, 1].sum() >= 0.05 else 0 for x in cats_datas.iloc[:, 1]]
#     # labels = [cats_datas.iloc[num, 0] if x > 0 else "" for num, x in enumerate(explode)]
#     ax.pie(cat_graph_datas.loc[:, "values"], explode= cat_graph_datas.loc[:, "explode"], 
#         labels = cat_graph_datas.loc[:, "labels"], autopct='%1.1f%%', startangle=90,
#         textprops= dict(color= "w", fontweight= "bold"), colors = cat_graph_datas.loc[:, "colors"])

#     plt.show()

#     # Récupérer les coordonnées gps des centres des états
#     map_datas["centroid"] = map_datas.centroid
#     map_datas["long"] = map_datas["centroid"].apply(lambda x: x.representative_point().coords[0][0])
#     map_datas["lat"] = map_datas["centroid"].apply(lambda x: x.representative_point().coords[0][1])

#     # Graphique
#     map_datas = map_datas.sort_values(by="id")
#     sum_mean_datas = sum_mean_datas.sort_values(by="state")

#     sum_cols = [x for x in sum_mean_datas.columns if re.search(r".*_sum", x)]
#     mean_cols = [x for x in sum_mean_datas.columns if re.search(r".*_mean", x)]
#     q_cols = [x for x in sum_mean_datas.columns if re.search(r".*_quant", x)]

#     for name, (sum_col, mean_col, q_col) in enumerate(zip(
#         sum_cols, mean_cols, q_cols)):

#         fig, ax = plt.subplots(figsize=(20, 10))

#         map_datas["geometry"].plot(ax= ax, cmap= "copper", alpha= 0.5)
        
#         scatter = ax.scatter(
#             map_datas["long"], 
#             map_datas["lat"], 
#             c = to_log(sum_mean_datas[sum_col]), 
#             cmap = "gist_rainbow", 
#             marker = "o", 
#             s = sum_mean_datas[q_col] *500, 
#             alpha = 0.6, 
#             )
#         handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
#         new_labels = get_label(sum_mean_datas[q_col], quantiles, mean_col, sum_mean_datas[mean_col])
#         labels = []
#         for i in range(len(new_labels)):
#             labels.append(f"$\\mathdefault{{{new_labels[i]}}}$")
#         if name > 0:
#             ax.legend(handles, labels, loc="lower left", title= "Moyenne", prop={"size": 25})
#             cbar = fig.colorbar(scatter)
#             cbar.set_label("Nb total: (log)")
#         else:
#             ax.legend(handles, labels, loc="lower left", title= "Nb jours depuis dernier achat moy", prop={"size": 25})
#             cbar = fig.colorbar(scatter)
#             cbar.set_label("Nb clients: (log)")

        
#         plt.title(graph_name[name])

#         for idx, row in map_datas.iterrows():
#             plt.text(row["long"], row["lat"], 
#                 s= row['id'], 
#                 horizontalalignment= "center", 
#                 fontweight = "bold")

#         plt.xlim(-90, -30)
#         plt.show()


# def get_segments(datas, conditions, sum_col, mean_col, graph_name, segment_names):
#     """data => dataset to make analysis on;
#     conditions => condition to get cluster (ex: datas["labels"] == 0);
#     sum_col => list of features to be summed
#     mean_col => list of features to be meaned
#     graph_name => list of names of the graphs
#     segment_names => list of names of segments or clusters
#     """
#     # Get map datas
#     map_datas = gpd.read_file("./archive (1)/brazil_geo.json")

#     # Boucle sur les segments
#     for condition, seg_name in zip(conditions, segment_names):
#         print(f"{'#' * 80}")
#         print(f"{seg_name}\n")

#         # Extraction des segments
#         segment = datas[condition]
#         # print(segment)

#         # Récupération des états
#         states_cols, cats_cols = get_states(segment)

#         # Transformation des données
#         sum_mean_datas, quantiles, cats_datas = sum_mean(cats_cols, states_cols, sum_col, mean_col, graph_name, segment)
#         # print(cats_datas)

#         # Graphique
#         print_graph(map_datas, cats_datas, sum_mean_datas, quantiles, graph_name)
    
    