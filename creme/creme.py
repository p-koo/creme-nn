import numpy as np
import shuffle
from tqdm import tqdm

############################################################################################
# TSS Context Dependence Test
############################################################################################

def context_dependence_test(model, x, tile_pos, num_shuffle, mean=True):
    """
    This test places a sequence pattern bounded by start and end in shuffled 
    background contexts -- in line with a global importance analysis. 

    Parameters
    ----------
        model : keras.Model 
            A keras model.
        x : np.array
            Single one-hot sequence shape (L, A).
        tile_pos : list
            List with start index and end index of pattern-of-interest along L (i.e. [start, end]).
        num_shuffle : int
            Number of shuffles to apply and average over.
        mean : bool
            If True, return the mean predictions across shuffles, otherwise return full predictions.

    Returns
    -------
        np.array : prediction of wild type sequence.
        np.array : prediction of mutant sequences.
    """

    # get wild-type prediction
    pred_wt = model.predict(x[np.newaxis])

    # crop pattern of interest
    start, end = tile_pos
    x_pattern = x[start:end,:]

    # loop over shuffles
    pred_mut = []
    for n in range(num_shuffle):
        x_mut = shuffle.dinuc_shuffle(x)
        x_mut[start:end,:] = x_pattern
        pred_mut.append(model.predict(x_mut)[np.newaxis][0])
    pred_mut = np.concatenate(pred_mut, axis=0)

    if mean:
        return pred_wt[0], np.mean(pred_mut, axis=0), np.std(pred_mut, axis=0)
    else:
        return pred_wt, pred_mut



############################################################################################
# TSS Context Swap Test
############################################################################################

def context_swap_test(model, x_source, x_target, tile_pos):
    """
    This test places a source sequence pattern bounded by start and end in a 
    target sequence context -- in line with a global importance analysis. 

    inputs:
        model: keras model 
            A keras model.
        x_source : np.array
            Source sequence (one-hot with shape (L, A) from which a pattern will be taken.
        x_targets : np.array
            Target seqeunces with shape (N,L,A) that will inherit a source pattern.
        tile_pos : list
            List of start and end index of pattern along L.
        mean : bool
            If True, return the mean predictions across shuffles, otherwise return full predictions.

    Returns
    -------
        np.array : prediction of wild type sequence.
        np.array : prediction of mutant sequences.
    """
    
    if len(x_target.shape) == 2:
        x_target = np.expand_dims(x_target, axis=0)
    if len(x_source.shape) == 2:
        x_source = np.expand_dims(x_source, axis=0)
    # get wild-type prediction
    

    start, end = tile_pos

    # loop through target sequences

    x_mut = np.copy(x_target)

    # place source pattern in target sequence
    x_mut[:,start:end,:] = x_source[:,start:end,:]

    # predict mutant sequence
    pred_mut = model.predict(x_mut)


    return pred_mut



############################################################################################
# CRE Necessity Test
############################################################################################

def generate_tile_shuffles(x, tile_set, num_shuffle):
    seq_mut = np.empty(([num_shuffle] + list(x.shape)))
    for n in range(num_shuffle):
        x_mut = np.copy(x)  # start a new mutant
        for pos in tile_set:  # tile set can include more than one start
            start, end = pos

            # shuffle tile
            x_mut[start:end, :] = shuffle.dinuc_shuffle(x_mut[start:end, :])

            # save mutant
            seq_mut[n] = x_mut
    return seq_mut

def necessity_test(model, x, tiles, num_shuffle, mean=True, return_seqs=False):
    """
    This test systematically measures how tile shuffles affects model predictions. 

    Parameters
    ----------
        model : keras.Model 
            A keras model.
        x : np.array
            Single one-hot sequence shape (L, A).
        tiles : list
            List of tile positions (start, end) to shuffle (i.e. [[start1, end1], [start2, end2],...]).
        num_shuffle : int
            Number of shuffles to apply and average over.
        mean : bool
            If True, return the mean predictions across shuffles, otherwise return full predictions.

    Returns
    -------
        np.array : prediction of wild type sequence.
        np.array : prediction of mutant sequences.
    """

    # get wild-type prediction
    pred_wt = model.predict(x[np.newaxis])

    # loop over shuffle positions list
    pred_mut = []
    all_muts = np.empty((len(tiles), num_shuffle, x.shape[-2], x.shape[-1]))
    for tile_i, pos in tqdm(enumerate(tiles), total=len(tiles)):
        start, end = pos

        # loop over number of shuffles
        pred_shuffle = []

        for n in range(num_shuffle):
            x_mut = np.copy(x)

            # shuffle tile
            x_mut[start:end,:] = shuffle.dinuc_shuffle(x_mut[start:end,:])
            all_muts[tile_i, n, :, :] = x_mut
            # predict mutated sequence
            pred_shuffle.append(model.predict(x_mut[np.newaxis])[0])
        pred_mut.append(pred_shuffle)
    pred_mut = np.array(pred_mut)


    if mean:
        test_res = [pred_wt, np.mean(pred_mut, axis=1), np.std(pred_mut, axis=1)]
    else:
        test_res = [pred_wt, pred_mut]
    if return_seqs:
        test_res.append(all_muts)
    return test_res



############################################################################################
# CRE Sufficiency Test
############################################################################################

def sufficiency_test(model, x, tss_tile, tiles, num_shuffle, tile_seq=None, mean=True, return_seqs=False):
    """
    This test measures if a region of the sequence is sufficient for model predictions. 

    Parameters
    ----------
        model : keras.Model 
            A keras model.
        x : np.array
            Single one-hot sequence shape (L, A).
        tss_tile : list
            List of the tss_tile position to include in all sufficiency test, i.e. [start, end].
        tiles : list
            List of tile positions (start, end) to include in sufficiency test (i.e. [[start1, end1], [start2, end2],...]).
        num_shuffle : int
            Number of dinuc shuffles to apply to sequence context and average over.
        mean : bool
            If True, return the mean predictions across shuffles, otherwise return full predictions.

    Returns
    -------
        np.array : prediction of wild type sequence.
        np.array : prediction of mutant sequences (dinuc shuffled sequence with TSS and tile).
        np.array : prediction of control sequence (dinuc shuffled sequence with TSS only).
    """

    # get wild-type prediction
    pred_wt = model.predict(x[np.newaxis])

    # loop over number of shuffles
    pred_mut = []
    pred_control = []
    for pos in tiles:
        start, end = pos

        pred_mut_shuffle = []
        pred_control_shuffle = []
        sequences = np.empty((num_shuffle, model.seq_length, 4))
        for n in range(num_shuffle):
            x_mut = shuffle.dinuc_shuffle(x)
            
            # embed tss tile
            x_mut[tss_tile[0]:tss_tile[1],:] = x[tss_tile[0]:tss_tile[1],:] 
            sequences[n] = x_mut.copy()
            # predict shuffled context with just TSS
            pred_control_shuffle.append(model.predict(x_mut[np.newaxis])[0])

            # embed tile of interest in
            if tile_seq:
                x_mut[start:end, :] = tile_seq
            else:
                x_mut[start:end,:] = x[start:end,:]

            # predict mutated sequence
            pred_mut_shuffle.append(model.predict(x_mut[np.newaxis])[0])

        # store results
        pred_mut.append(pred_mut_shuffle)
        pred_control.append(pred_control_shuffle)

    pred_mut = np.array(pred_mut)
    pred_control = np.array(pred_control)

    if mean:
        test_res = [pred_wt[0], np.mean(pred_mut, axis=1), np.std(pred_mut, axis=1), np.mean(pred_control, axis=1), np.std(pred_control, axis=1)]
    else:
        test_res = [pred_wt, pred_mut, pred_control]
    if return_seqs:
        test_res.append(sequences)
    return test_res


############################################################################################
# TSS-CRE Distance Test
############################################################################################

def distance_test(model, x, tile_fixed_coord, tile_var_coord, test_positions, num_shuffle, mean=True, seed=False):
    """
    This test maps out the distance dependence of tile1 (anchored) and tile 2 (variable). 
    Tiles are placed in dinuc shuffled background contexts, in line with global importance analysis. 

    Parameters
    ----------
        model : keras.Model 
            A keras model.
        x : np.array
            Single one-hot sequence shape (L, A).
        tile_fixed_coord : list
            List with start index and end index of tile that is anchored (i.e. [start, end]).
        tile_var_coord : list
            List with start index and end index of tile that is to be tested at avilable_tiles.
        test_positions : list
            List with start index of positions to test tile_var.
        num_shuffle : int
            Number of shuffles to apply and average over.
        mean : bool
            If True, return the mean predictions across shuffles, otherwise return full predictions.

    Returns
    -------
        np.array : prediction of dinuc sequence with tiles placed in original locations.
        np.array : prediction of dinuc sequenc with tiles placed in variable locations.
    """

    # crop pattern of interest
    x_tile_fixed = x[tile_fixed_coord[0]:tile_fixed_coord[1],:]  # fixed tile sequence
    x_tile_var = x[tile_var_coord[0]:tile_var_coord[1],:]  # variable position tile sequence

    # get sufficiency of tiles in original positions
    pred_control = []
    for n in range(num_shuffle):
        # shuffle sequence and place tiles in respective positions
        if seed:
            x_mut = shuffle.dinuc_shuffle(x, seed=n)
        else:
            x_mut = shuffle.dinuc_shuffle(x)
        x_mut[tile_fixed_coord[0]:tile_fixed_coord[1],:] = x_tile_fixed
        x_mut[tile_var_coord[0]:tile_var_coord[1],:] = x_tile_var

        # predict mutant sequence
        pred_control.append(model.predict(x_mut[np.newaxis])[0])
    pred_control = np.array(pred_control)

    # loop over embedding tile_var in available position list
    pred_mut = []
    tile_len = tile_var_coord[1] - tile_var_coord[0]
    for start in tqdm(test_positions):

        # loop over number of shuffles
        pred_shuffle = []
        for n in range(num_shuffle):

            # shuffle sequence
            if seed:
                x_mut = shuffle.dinuc_shuffle(x, seed=n)
            else:
                x_mut = shuffle.dinuc_shuffle(x)

            # place tile 1 in original location
            x_mut[tile_fixed_coord[0]:tile_fixed_coord[1],:] = x_tile_fixed

            # place tile 2 in new position
            x_mut[start:start+tile_len,:] = x_tile_var

            # predict mutant sequence
            pred_shuffle.append(model.predict(x_mut[np.newaxis])[0])
        pred_mut.append(pred_shuffle)
    pred_mut = np.array(pred_mut)

    if mean:
        return np.mean(pred_control, axis=0), np.std(pred_control, axis=0), np.mean(pred_mut, axis=1), np.std(pred_mut, axis=1)
    else:
        return pred_control, pred_mut 



############################################################################################
# CRE Higher-order Interaction Test
############################################################################################






def higher_order_interaction_test(model, x, cre_tiles_to_test, optimization, num_shuffle=10, num_rounds=None):
    """
    This test performs a greedy search to identify which tile sets lead to optimal changes
    in model predictions. In each round, a new tile is identified, given the previous sets 
    of tiles. Effect size is measured by placing tiles in dinuc shuffled sequences.

    Parameters
    ----------
        model : keras.Model 
            A keras model.
        x : np.array
            Single one-hot sequence shape (L, A).
        fixed_tiles : list
            List with fixed tiles, each with a list that consists of start index and end index.
        available_tiles : list
            List with available tiles, each with a list that consists of start index and end index.
        num_shuffle : int
            Number of shuffles to apply and average over.
        num_rounds : int
            Number of rounds to perform greedy search.
        optimization : np.argmax or np.argmin
            Function that identifies tile index for each round of greedy search.
        reduce_fun : np.mean
            Function to reduce (multivariate) predictions to a scalar.

    Returns
    -------
        np.array : prediction of wild type sequence.
        np.array : prediction of mutant sequences in each optimization round.
        list : list of fixed tiles from each round.
    """

    result_summary = {}
    if not num_rounds:
        num_rounds = len(cre_tiles_to_test)

    for iteration_i in tqdm(range(num_rounds)):
        result_summary[iteration_i] = {}
        # run one sweep of tile shuffles and keep shuffled seqs
        pred_wt, pred_mut, all_muts = necessity_test(model, x, cre_tiles_to_test, num_shuffle, False, True)

        # get per tile predictions (average across bins)
        per_tile_preds = pred_mut[..., 0].mean(-1)  # [tile number, shuffle n]

        result_summary[iteration_i]['initial_pred'] = pred_wt.mean() # keep track of initial seq prediction
        result_summary[iteration_i]['preds'] = per_tile_preds  # save all tile preds for comparing to hypothetical model
        per_tile_mean = per_tile_preds.mean(axis=-1) # average across shuffles

        # find optimal tile
        selected_tile_i = optimization(per_tile_mean)  # find best tile index
        best_tile_preds = per_tile_preds[selected_tile_i, :] # all shuffle outputs for best tile
        keep_shuffled = cre_tiles_to_test[selected_tile_i] # tile coords of the best tile
        result_summary[iteration_i]['selected_tile'] = keep_shuffled # keep record
        cre_tiles_to_test.remove(keep_shuffled) # remove this for next iteration

        selected_mean_pred = per_tile_mean[selected_tile_i] # select the best tile prediction for trace
        result_summary[iteration_i]['selected_mean_pred'] = selected_mean_pred
        # update seq for next iteration selecting sequence yielding the closest prediction to mean
        x = all_muts[selected_tile_i, np.argmin(np.abs(best_tile_preds - selected_mean_pred)), :, :]
    return result_summary

############################################################################################
# CRE Multiplicity Test
############################################################################################
def multiplicity_test(model, x, tss_tile_coord, cre_tile_coord, cre_tile_seq, test_coords, num_shuffle, num_copies,
                      optimization):
    """

    :param model:
    :param x:
    :param tss_tile_coord:
    :param cre_tile_coord:
    :param cre_tile_seq:
    :param test_coords:
    :param num_shuffle:
    :param num_copies:
    :return:
    """
    tile_positions = test_coords.copy()
    max_per_iter = []
    best_tiles = []

    for _ in tqdm(range(num_copies)):
        # loop over number of shuffles
        normalized_preds = np.empty((num_shuffle, len(tile_positions)))
        for n in range(num_shuffle):
            x_mut = shuffle.dinuc_shuffle(x)  # destroy all
            # embed fixed tiles (TSS + fixed CREs)
            x_mut[tss_tile_coord[0]:tss_tile_coord[1], :] = x[tss_tile_coord[0]:tss_tile_coord[1], :]  # add TSS
            x_mut_control = x_mut.copy()
            x_mut_control[cre_tile_coord[0]: cre_tile_coord[1]] = cre_tile_seq
            for fixed_tile in best_tiles:
                x_mut[fixed_tile[0]:fixed_tile[1], :] = x[fixed_tile[0]:fixed_tile[1],
                                                        :]  # use original sequence for fixed tiles

            # predict shuffled context with just TSS + fixed CRE tiles
            pred_control = model.predict(x_mut_control).mean()

            # systematically test CRE effect at different positions
            for t, (start, end) in enumerate(tile_positions):
                x_mut_pos = x_mut.copy()
                x_mut_pos[start:end, :] = cre_tile_seq

                # predict mutated sequence
                pred_mut = model.predict(x_mut_pos).mean()
                normalized_preds[n, t] = pred_mut / pred_control
        normalized_preds_mean = normalized_preds.mean(axis=0)

        best_position_index = optimization(normalized_preds_mean)  # select based on mean
        best_position = tile_positions[best_position_index]
        tile_positions.remove(best_position)  # remove from test positions
        best_tiles.append(best_position)  # put in bag of best positions
        max_per_iter.append(normalized_preds[:, best_position_index])  # save all shuffle runs, not mean
    return {'max_per_iter': max_per_iter, 'best_tiles': best_tiles}



########################################################################################
# Normalization functions
########################################################################################


def context_effect_on_tss(pred_wt, pred_mut):
    """Normalization based on difference between the effect size of the mutation and wt divided by wt"""

    return (pred_wt - pred_mut) / pred_wt



def fold_change_over_control(pred_wt, pred_mut, bin_index):
    """Normalization based on difference between the effect size of the mutation and wt divided by wt"""
    if len(pred_mut.shape) == 1:
        return pred_mut[bin_index] / pred_wt[bin_index]
    else:
        return pred_mut[:,bin_index] / pred_wt[bin_index]


def normalized_tile_effect(pred_wt, pred_mut, pred_control, bin_index):
    """Normalization used for sufficiency test"""
    return (pred_mut[:,bin_index] - pred_control[:,bin_index])/pred_wt[bin_index]


def reduce_pred_index(pred, bin_index):
    """Reduce multivariate prediction by selecting an index"""
    return pred[:, bin_index]


def remove_tss_tile(tiles, tile_index):
    """Remove a tile form a list of tile coordinates."""
    del tiles[tile_index]



