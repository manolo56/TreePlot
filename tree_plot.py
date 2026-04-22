
import pandas as pd
import numpy as np
import numbers

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch


from Bio import Phylo
from Bio.Phylo.BaseTree import Tree
import collections
from copy import deepcopy
from cycler import cycler
import colorsys



class TreePlot:
    def __init__(self, tree, circ_offset=0, reverse_order=False):
        self.tree = deepcopy(tree)
        self.df_tree = self.get_dataframe_representation(circ_offset=circ_offset, reverse_order=reverse_order)
        self.display_depth = self.df_tree["Child_depth"].max()
        self.max_node_order = self.df_tree["Node_order"].max()
    
    def postorder_traverse(self):
        def dfs(elem):
            for v in elem.clades:
                yield from dfs(v)
            yield elem

        yield from dfs(self.tree.clade)

    def assign_node_index(self) :
        current_node = 0 
        for c in self.postorder_traverse() :
            c.node_index = current_node
            current_node += 1

    def rename_leaves(self, leaf_dic) :
        for c in self.postorder_traverse() :
            if c.name in leaf_dic :
                c.name = leaf_dic[c.name]

    def get_dataframe_representation(self, circ_offset=0, reverse_order=False) :
        # copy tree object and assign node index by post-order traversal
        self.assign_node_index()

        nodes = []
        Q = collections.deque([self.tree.clade]) 
        while Q :
            v = Q.popleft()
            for child in v.clades :
                nodes.append({"Parent" : v.node_index, "Child" : child.node_index, "Branch_length" : child.branch_length, 
                            "Name" : child.name if child.name else "", "Is_terminal" : len(child.clades) == 0})
            Q.extend(v.clades)
        nodes.append({"Parent" : len(nodes)+1, "Child" : len(nodes), "Branch_length" : 0, 
                      "Name" : "Root", "Is_terminal" : False}) # add root which is its own parent

        df_tree = pd.DataFrame(nodes).sort_values(by="Child").reset_index(drop=True) # very important to sort values

        df_tree["Node_order"] = -1.0 # initiate node order at -1
        df_tree.loc[df_tree["Is_terminal"],"Node_order"] = df_tree.loc[df_tree["Is_terminal"]].reset_index().index # set node order for leaves

        while (df_tree["Node_order"] == -1).any() : # traverse the tree and assign node order to all internal nodes including root
            for i, row in df_tree.iterrows():
                if row["Node_order"] != -1:
                    continue
                children = df_tree[df_tree["Parent"] == row["Child"]]
                if (children["Node_order"] != -1).all():
                    df_tree.loc[i, "Node_order"] = children["Node_order"].mean()
                    # the node order is the mean of the children's node order. Weight by subclade size ? No it's worse

        if reverse_order : 
            df_tree["Node_order"] = df_tree["Node_order"].max() - df_tree["Node_order"]
        df_tree["Theta"] = circ_offset +2*np.pi*df_tree["Node_order"]/(df_tree["Node_order"].max() + 1) - np.pi
        depths = self.tree.depths()
        depth_df = pd.DataFrame({
            "node_index": [c.node_index for c in depths.keys()] + [len(nodes)],
            "depth": list(depths.values()) + [0.0]
        }) # add the root's depth

        df_tree = df_tree.merge(
            depth_df.rename(columns={"node_index": "Child", "depth": "Child_depth"}),
            on="Child"
        ) # Add the depth of the node (viewed as a child)

        df_tree = df_tree.merge(
            depth_df.rename(columns={"node_index": "Parent", "depth": "Parent_depth"}),
            on="Parent"
        ) # Add the depth of the node's parent
        
        return df_tree
        
    
    def arc_to_lines(self, radius, theta_min, theta_max, n=30):
        theta = np.linspace(theta_min, theta_max, n)
        return np.column_stack([radius*np.cos(theta), radius*np.sin(theta)])

    def assign_branch_lw(self, branch_lw) : # see if we keep, now that we changed the drawing function it is not possible to assign a different width to a single branch
        if branch_lw is None :
            self.df_tree["Branch_lw"] = 1
        elif isinstance(branch_lw, numbers.Real) :
            if branch_lw<= 0 :
                raise ValueError("branch_lw should be a positive value if numeric")
            else :
                self.df_tree["Branch_lw"] = branch_lw
        elif isinstance(branch_lw, dict):
                def format_branch_lw(clade):
                    return branch_lw.get(clade, 1)
        else :
            if not callable(branch_lw):
                raise TypeError("branch_lw must be either a dict or a callable (function)")
            format_branch_lw = branch_lw 
            self.df_tree["Branch_lw"] = [format_branch_lw(name) for name in self.df_tree["Name"].values]


    def draw_tree(self, show_nodes, node_size, node_color, node_edge_color, edge_lw) : # new drawing function 
        order_col = "Theta" if self.how == "circular" else "Node_order"
        shapes, linewidths = [], []
        children_map = self.df_tree.groupby("Parent").indices
        node_X, node_Y = [], []
        for row in self.df_tree.itertuples():
            if not row.Is_terminal:
                child_idx = children_map.get(row.Child, [])
                children = self.df_tree.iloc[child_idx].sort_values(by=order_col)
                linewidths.append(children["Branch_lw"].min())
                if self.how == "circular" :
                    node_x = row.Child_depth*np.cos(row.Theta)
                    node_y = row.Child_depth*np.sin(row.Theta)
                    child1_x = children["Child_depth"].values[0]*np.cos(children["Theta"].values[0])
                    child1_y = children["Child_depth"].values[0]*np.sin(children["Theta"].values[0])
                    child2_x = children["Child_depth"].values[1]*np.cos(children["Theta"].values[1])
                    child2_y = children["Child_depth"].values[1]*np.sin(children["Theta"].values[1])
                    order_line = self.arc_to_lines(row.Child_depth, children["Theta"].values[0], children["Theta"].values[1])
                    new_shape = np.vstack([ [[child1_x, child1_y]], order_line, [[child2_x, child2_y]]])
                    shapes.append(new_shape)
                    if row.Child_depth > 0 : # don't draw the root
                        node_X.append(node_x)
                        node_Y.append(node_y)
                elif self.how=='linear' :
                    node_x = row.Child_depth
                    node_y = row.Node_order
                    child1_x = children["Child_depth"].values[0]
                    child1_y = children["Node_order"].values[0]
                    child2_x = children["Child_depth"].values[1]
                    child2_y = children["Node_order"].values[1]
                    new_shape = [(child1_x, child1_y), (row.Child_depth, child1_y), (row.Child_depth, child2_y), (child2_x, child2_y)] 
                    shapes.append(new_shape)
                    if row.Child_depth > 0 : # don't draw the root
                        node_X.append(node_x)
                        node_Y.append(node_y)
        
        lc = LineCollection(shapes,
                            linewidths=linewidths,
                            colors='black',
                            capstyle='butt',
                            joinstyle='miter'
                        )
        self.ax.add_collection(lc)

        if show_nodes :
            self.ax.scatter(node_X, node_Y,s=node_size, color=node_color, edgecolors=node_edge_color, lw=edge_lw, zorder=2)
        
        return 


    def set_view(self, margin=0.05) :
        self.margin = margin
        if self.how == 'circular' :
            sidelim = self.display_depth * (1 + margin)
            self.ax.set_xlim([-sidelim, sidelim])
            self.ax.set_ylim([-sidelim, sidelim])
            self.ax.set_aspect('equal')
        else :
            self.ax.set_xlim([-self.display_depth*margin, self.display_depth*(1+margin)])
            self.ax.set_ylim([-self.max_node_order*margin, self.max_node_order*(1+margin)])
        self.ax.axis('off')

            

    def plot_tree(self, how='circular', fig = None, ax = None, figsize=None,
                    branch_lw = None, show_nodes=False, node_size=5, node_color='black', node_edge_color='white', edge_lw=0.1,
                     margin=0.05) :
        if fig is None or ax is None:
            if ax is not None :
                print("Ignoring ax argument as fig argument is set to None")
            if fig is not None :
                print("Ignoring fig argument as ax argument is set to None")
            if figsize is None :
                figsize = (8,8)
            fig, ax = plt.subplots(figsize=figsize)

        self.fig = fig
        self.ax = ax
        self.how = how
        
        self.assign_branch_lw(branch_lw)
        self.draw_tree(show_nodes, node_size, node_color, node_edge_color, edge_lw)
        self.set_view(margin)

        return


    def add_scale(self, fontsize=12, lw=None, x=None, y=None, unit=None) :
        xlims = self.ax.get_xlim()
        ylims = self.ax.get_ylim()
        max_height = self.df_tree["Child_depth"].max()

        if x is None :
            x = 0.1
        if y is None :
            y = 0
        if lw is None :
            if "Branch_lw" in self.df_tree.columns :
                lw = self.df_tree["Branch_lw"].mean()
            else :
                lw = 1
        p0 = self.ax.transAxes.transform((x, y))
        x0_data, y0_data = self.ax.transData.inverted().transform(p0)
        
        log_val = int(np.floor(np.log10(max_height / 10)))
        line_length = 10**log_val
        self.ax.plot([x0_data, x0_data + line_length], [y0_data, y0_data], color='black', lw=lw, clip_on=False)
        if np.abs(log_val) >= 3 : 
            legend = fr"  $10^{{{log_val}}}$"
        elif log_val < 0 :
            legend = "  " + str(line_length)
        else :
            legend = "  " + str(int(line_length))
        if unit :
            legend = legend + " " + unit
        self.ax.text(x0_data + line_length, y0_data, s=legend, fontsize=fontsize, ha='left', va='center', clip_on=False)

        self.ax.set_xlim(xlims)
        self.ax.set_ylim(ylims)

        return 
        
            


    def add_tip_labels(self, fontfamily='sans-serif', tip_offset=0.01, fontsize=None, label_func=None, margin=None,
                       align_labels=False, aln_lw=0.5, aln_ls='--', aln_color='grey', circ_padding=0.02, **kwargs) :
        self.df_tree["Label"] = self.df_tree["Name"]
        is_term = self.df_tree["Is_terminal"]
        if label_func :
            self.df_tree.loc[is_term, "Label"] = self.df_tree.loc[is_term, "Name"].apply(label_func)
        
        tips = self.df_tree[self.df_tree["Is_terminal"]].copy()
        # tips["depth_label"] = tips["Child_depth"] + tip_offset * (xlims[-1]-xlims[0])
        max_height = tips["Child_depth"].max()
        length_offset = tip_offset * max_height # tip_offset in percentage of the current display length (should be equal to tree depth)

        # determine labels coordinates and joining lines coordinate if necessary
        if self.how == 'circular' :
            if not align_labels :
                tips["x"] = (tips["Child_depth"] + length_offset) * np.cos(tips["Theta"])
                tips["y"] = (tips["Child_depth"] + length_offset) * np.sin(tips["Theta"])
            else : 
                tips["line_start_x"] = (tips["Child_depth"] + length_offset / 2)* np.cos(tips["Theta"]) 
                tips["line_start_y"] = (tips["Child_depth"] + length_offset / 2) * np.sin(tips["Theta"]) 
                tips["line_stop_x"] = (max_height + length_offset / 2)* np.cos(tips["Theta"]) 
                tips["line_stop_y"] = (max_height + length_offset / 2) * np.sin(tips["Theta"]) 
                tips["x"] = (max_height + length_offset) * np.cos(tips["Theta"]) 
                tips["y"] = (max_height + length_offset) * np.sin(tips["Theta"]) 
        elif self.how == 'linear' :
            if not align_labels :
                tips["x"] = tips["Child_depth"] + length_offset
                tips["y"] = tips["Node_order"]
            else :
                tips["line_start_x"] = tips["Child_depth"] + length_offset / 2
                tips["line_start_y"] = tips["Node_order"]
                tips["line_stop_x"] = max_height + length_offset / 2 
                tips["line_stop_y"] = tips["Node_order"]
                tips["x"] = max_height + length_offset
                tips["y"] = tips["Node_order"]
        if fontsize is None :
            _, fig_h = self.fig.get_size_inches()
            pos = self.ax.get_position()
            height_inch_units = pos.height*fig_h
            if self.max_node_order > 50 :
                fontsize = height_inch_units / self.max_node_order * 100
            else :
                fontsize = height_inch_units * 2 # fixed fontsize in points
        txt_list = []
        line_list = []
        for row in tips.itertuples():
            if self.how == 'circular' :
                angle = row.Theta
            elif self.how == 'linear' :
                angle = 0
            rotation = np.degrees(angle)
            if np.pi/2 < np.radians(rotation % 360) < 3*np.pi/2:
                rotation += 180
                ha = "right"
            else:
                ha = "left"
            
            txt = self.ax.text(row.x, row.y, row.Label,
                    rotation=rotation, rotation_mode='anchor',
                    ha=ha, va='center', fontsize=fontsize,
                    fontfamily=fontfamily, transform=self.ax.transData)
            txt_list.append(txt)

            if align_labels :
                line_list.append([(row.line_start_x, row.line_start_y), (row.line_stop_x, row.line_stop_y)])
        
        if align_labels :
            self.ax.add_collection(LineCollection(line_list, linewidths=aln_lw, linestyle=aln_ls, color=aln_color, **kwargs)) 

        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        if self.how == 'linear' :
            max_depth = max_height + length_offset
            for txt in txt_list :
                bbox = txt.get_window_extent(renderer=renderer)
                inv = self.ax.transData.inverted()
                bbox_data = bbox.transformed(inv)
                new_depth = bbox_data.x1
                max_depth = max(max_depth, new_depth)                
            
            # self.ax.set_xlim([xlims[0], max_depth])
            
        else :
            max_depth = max_height + length_offset
            for txt in txt_list :
                bbox = txt.get_window_extent(renderer=renderer)
                inv = self.ax.transData.inverted()
                bbox_data = bbox.transformed(inv)
                corners = [
                    (bbox_data.x0, bbox_data.y0),
                    (bbox_data.x0, bbox_data.y1),
                    (bbox_data.x1, bbox_data.y0),
                    (bbox_data.x1, bbox_data.y1),
                ]
                max_r = max(np.hypot(x, y) for x, y in corners)
                max_depth = max(max_depth, max_r)
            max_depth *= (1 + circ_padding) # padding for circular plot

        self.display_depth = max_depth
        if margin is not None :
            self.set_view(margin=margin)
        else :
            self.set_view(margin=self.margin)
        
        return
    
    def ggplot2_clone(self, n, h=(15, 375), l=0.65, s=0.90):
        """
        Approximate ggplot2::scale_color_hue()
        using evenly spaced hues.
        """
        hues = np.linspace(h[0], h[1], n + 1)[:-1] / 360.0
        return [colorsys.hls_to_rgb(hh % 1, l, s) for hh in hues]


    def add_genome_properties(self, properties_df, prop_color_df = None, name_col="Name", default_color_map = None,
                          fontsize=12, prop_width = 0.02, margin=0.05, offset=0.02,
                          legend_spacing=0.02, legend_fontsize=12, legend_margin=0.0) :
        tips = self.df_tree[self.df_tree["Is_terminal"]].copy()
        
        if not default_color_map :
            get_color_list = lambda n : self.ggplot2_clone(n)
        else :
            get_color_list = lambda n : list(default_color_map(np.linspace(0,1,n)))

        if not name_col in properties_df :
            raise KeyError(f"Could not find columns {name_col} in properties_df")
        properties_df = properties_df.set_index(name_col).convert_dtypes("numpy_nullable")
        prop_values = properties_df.melt(var_name="Property", value_name="Value").drop_duplicates().reset_index(drop=True)

        if prop_color_df is not None and len(prop_color_df) and ("Property" in prop_color_df.columns
            and "Value" in prop_color_df.columns and "Color" in prop_color_df.columns) :
            prop_values = prop_values.merge(prop_color_df[["Property", "Value", "Color"]], 
                                            on=["Property", "Value"], how='left')
        else :
            prop_values["Color"] = pd.NA
        prop_values = prop_values.loc[~prop_values["Value"].isna()] 
        no_color = prop_values["Color"].isna()
        colors = get_color_list(no_color.sum())
        prop_values.loc[no_color, "Color"] = pd.Series(colors, index=no_color[no_color].index)


        polygons, colors = [], []
        tips = tips.merge(properties_df.rename_axis("Name").reset_index(), on="Name", how='left')

        old_display_depth = self.display_depth
        self.display_depth *= (1 + (len(properties_df.columns)+1)*offset + len(properties_df.columns)*prop_width) 
        self.set_view(margin=margin)


        for i_prop, prop in enumerate(properties_df.columns) :
            depth1 = old_display_depth*(1 + (i_prop+1)*offset + i_prop*prop_width) 
            depth2 = old_display_depth*(1 + (i_prop+1)*offset + (i_prop+1)*prop_width) 
            groups, current_group = [], [0]
            prop_tips = tips.reset_index(drop=True)
            prop_tips = prop_tips.loc[~prop_tips[prop].isna()]
            for index_tip in prop_tips.index[1:]:
                if (index_tip-1 in prop_tips.index) and tips.loc[index_tip,prop] == tips.loc[index_tip-1,prop] :
                    current_group.append(index_tip)
                else:
                    groups.append(current_group)
                    current_group = [index_tip]
            groups.append(current_group)

            if self.how =='linear' : # add legends on top
                self.ax.text(depth1 + old_display_depth*prop_width/2, self.max_node_order+1/2, "  "+prop, 
                             ha='left', va='center', clip_on=False,
                             rotation=60, rotation_mode='anchor', fontsize=fontsize)
            for g in groups :
                prop_val = prop_tips.loc[g[0],prop]
                if self.how == 'circular' :
                    thetas = tips.iloc[min(g):(max(g)+1)]["Theta"].values
                    theta1 = min(thetas) - 1/2 * 2*np.pi / (len(tips) + 1)
                    theta2 = max(thetas) + 1/2 * 2*np.pi / (len(tips) + 1)
                    angles = np.linspace(theta1, theta2, 100)
                    outer = np.column_stack([
                        depth2 * np.cos(angles),
                        depth2 * np.sin(angles)
                    ])
                    inner = np.column_stack([
                        depth1 * np.cos(angles[::-1]),
                        depth1 * np.sin(angles[::-1])
                    ])
                    poly = np.vstack([outer, inner])
                elif self.how == 'linear' :
                    orders = tips.iloc[min(g):(max(g)+1)]["Node_order"].values
                    order1 = min(orders) - 1/2 
                    order2 = max(orders) + 1/2 
                    poly = [[depth2, order1], [depth2, order2], [depth1, order2], [depth1, order1]]
                
                polygons.append(poly)
                colors.append(prop_values.loc[(prop_values["Property"]==prop)&(prop_values["Value"]==prop_val), 
                                              "Color"].values[0]
                              )
        collection = PolyCollection(polygons,
                                    facecolors=colors,
                                    edgecolors='none',
                                    clip_on=False)
        self.ax.add_collection(collection)
        

        ##### Legend : 
        self.fig.canvas.draw() # maybe not needed ?
        
        def make_property_legend(color_dict):
            handles = []
            for key, color in color_dict.items():
                handles.append(Patch(facecolor=color, edgecolor='none', label=str(key)))
            return handles
        
        def extract_bbox(fig, leg) :
            fig.canvas.draw()
            bbox = leg.get_window_extent()
            inv = fig.transFigure.inverted()
            bbox_fig = bbox.transformed(inv)
            return bbox_fig

        # Get the top right and bottom left positions in figure coordinates
        if self.how == 'circular' :
            # Top right
            x_disp, y_disp = self.ax.transData.transform((self.display_depth, self.display_depth))
            x_fig, y_fig = self.fig.transFigure.inverted().transform((x_disp, y_disp))

            # Bottom left
            x_disp2, y_disp2 = self.ax.transData.transform((-self.display_depth, -self.display_depth))
            x_fig2, y_fig2 = self.fig.transFigure.inverted().transform((x_disp2, y_disp2))

        else :
            # Top right
            x_disp, y_disp = self.ax.transData.transform((self.display_depth, self.max_node_order))
            x_fig, y_fig = self.fig.transFigure.inverted().transform((x_disp, y_disp))

            # Bottom left
            x_disp2, y_disp2 = self.ax.transData.transform((0, 0))
            x_fig2, y_fig2 = self.fig.transFigure.inverted().transform((x_disp2, y_disp2))


        # Define drawing positions
        # constant positions :
        y_topline = y_fig - legend_margin * (y_fig - y_fig2)
        y_bottom_line = y_fig2 + legend_margin * (y_fig - y_fig2)
        x_leftline = x_fig + legend_spacing*(x_fig - x_fig2) # initial position of legend
        # moving positions
        x_position = x_leftline 
        y_position = y_topline
        x_right = x_position # current right-most bbox boundary
        # estimated only once
        estimated_ratio = 0
        
        for i_prop, prop in enumerate(properties_df.columns) :
            handles  = make_property_legend(prop_values[prop_values["Property"]==prop].set_index("Value").to_dict()["Color"])
            num_objects = len(handles) + 2 # title ~ 2 handles
            
            if estimated_ratio > 0 : # already a legend block plotted before : 
                estimated_height = estimated_ratio * num_objects
                if y_position - estimated_height > y_bottom_line : # still space in this column
                    pass
                else : # no space left : start at a new column
                    x_position = x_right + legend_spacing*(x_fig - x_fig2)
                    y_position = y_topline
                    x_right = x_position
                            
            leg = self.fig.legend(handles, [l.get_label() for l in handles], title=prop,
                    loc="upper left", bbox_to_anchor=(x_position, y_position),
                    fontsize=legend_fontsize, title_fontsize=legend_fontsize+2,
                    frameon=False)
            
            # Updated new position of putative next legend block and the remaining space in this column
            bbox = extract_bbox(self.fig, leg)
            if estimated_ratio == 0 : # first legend block : compute the label number to space taken ratio
                estimated_ratio = (bbox.y1 - bbox.y0) / (len(handles) + 2) # title ~ 2 handles        
            y_position = bbox.y0 - legend_spacing
            x_right = max(x_right, bbox.x1)
        return
            


