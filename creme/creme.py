import numpy as np
import shuffle
from tqdm import tqdm
import operator


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
    shuffled_seqs = shuffle.dinuc_shuffle(x, num_shuffle)  # destroy all
    shuffled_seqs[:, tss_tile_coord[0]:tss_tile_coord[1], :] = x[tss_tile_coord[0]:tss_tile_coord[1], :].copy()

    only_tss_pred = model.predict(shuffled_seqs).mean()

    tss_and_cre = shuffled_seqs.copy()
    tss_and_cre[:, cre_tile_coord[0]: cre_tile_coord[1]] = cre_tile_seq

    tss_and_cre_pred = model.predict(tss_and_cre).mean()

    tile_positions_to_test = test_coords.copy()
    current_seq_version = shuffled_seqs.copy()
    all_mutants = []
    best_tss_signal = []
    selected_tile_order = []
    for _ in tqdm(range(num_copies)):
        test_seqs = np.empty((num_shuffle, len(tile_positions_to_test), model.seq_length, 4))
        mutant_preds = np.empty((num_shuffle, len(tile_positions_to_test)))
        for s, shuffled_seq in enumerate(current_seq_version):
            for t, (tile_start, tile_end) in enumerate(tile_positions_to_test):
                test_seq = shuffled_seq.copy()
                test_seq[tile_start: tile_end] = cre_tile_seq.copy()
                mutant_preds[s, t] = model.predict(test_seq).mean()
                test_seqs[s, t, ...] = test_seq.copy()
        best_index = optimization(mutant_preds.mean(axis=0))
        selected_tile = tile_positions_to_test[best_index]
        best_tss_signal.append(mutant_preds.mean(axis=0)[best_index])
        tile_positions_to_test.remove(selected_tile)
        selected_tile_order.append(selected_tile)
        all_mutants.append(mutant_preds)
        current_seq_version = test_seqs[:, best_index, ...].copy()
    return {'only_tss_pred': only_tss_pred, 'tss_and_cre_pred': tss_and_cre_pred, 'best_tss_signal': best_tss_signal,
            'selected_tile_order': selected_tile_order, 'all_mutants': all_mutants}

########################################################################################
# Pruning function
########################################################################################



def prune_sequence(model, wt_seq, control_sequences, mut, whole_tile_start, whole_tile_end, scales, thresholds, frac,
                   N_batches, cre_type='enhancer'):
    remove_tiles = []

    # save what to put back from wt sequence in form of coordinates
    insert_coords = [[whole_tile_start, whole_tile_end]]
    pruned_seqs = control_sequences.copy()
    bps = np.zeros((whole_tile_end - whole_tile_start))
    result_summary = {}

    for (window, threshold, N_batch) in zip(scales, thresholds, N_batches):
        result_summary[window] = {'scores': [], 'bps': []}
        print(f"Tile size = {window}, threshold = {threshold}")

        step = int(window * frac)
        test_coords = []

        for insert_coord in insert_coords:
            pruned_seqs[:, insert_coord[0]: insert_coord[1], :] = wt_seq[insert_coord[0]: insert_coord[1]].copy()
            bps[insert_coord[0] - whole_tile_start: insert_coord[1] - whole_tile_start] = 1  # count added bps
            test_coords += [[s, s + window] for s in list(range(insert_coord[0], insert_coord[1] - step + 1, step))]

        score = model.predict(pruned_seqs).mean() / mut
        print(f"Starting score: {score}")

        test_coords = np.array(test_coords)
        print(len(test_coords))

        final_check_seq = pruned_seqs.copy()
        all_removed_tiles = np.array([[], []]).T

        print("Starting optimization...")

        if cre_type == 'enhancer':
            comp = operator.gt
        elif cre_type == 'silencer':
            comp = operator.lt


        while comp(score, threshold) and len(test_coords):

            print(f'score = {score}')

            pruned_seqs = final_check_seq.copy()  # save removed seq tiles
            new_test_coords = []
            for test_coord in test_coords:
                if test_coord not in all_removed_tiles:
                    new_test_coords.append(test_coord)
            test_coords = new_test_coords
            print(f"Number of tiles to test: {len(test_coords)}")
            results = []
            for test_coord in test_coords:
                # remove subtile
                test_seqs = pruned_seqs.copy()  # don't change pruned_seqs yet
                test_seqs[:, test_coord[0]: test_coord[1], :] = control_sequences[:, test_coord[0]: test_coord[1],
                                                                :].copy()
                results.append(model.predict(test_seqs).mean())

            if cre_type == 'enhancer': # prune out silencers, ie. tiles that when shuffled lead to higher pred
                remove_tiles = np.array(test_coords)[np.argsort(results)[-N_batch:]]  # choose N useless
            elif cre_type == 'silencer': # remove enhancers = tiles the shuffling of which leads to drop in TSS
                remove_tiles = np.array(test_coords)[np.argsort(results)[:N_batch]]  # choose N useless

            all_removed_tiles = np.concatenate([all_removed_tiles, remove_tiles])  # add to santa's bad list

            # final check
            final_check_seq = pruned_seqs.copy()
            for tile in remove_tiles:
                final_check_seq[:, tile[0]: tile[1], :] = control_sequences[:, tile[0]: tile[1],
                                                          :].copy()  # prune out selected tiles
                bps[tile[0] - whole_tile_start: tile[1] - whole_tile_start] = 0

            score = model.predict(final_check_seq).mean() / mut
            print(f"Number of tiles at the end of iteration: {len(test_coords)}, score = {score}, bps = {bps.sum()}")
            result_summary[window]['scores'].append(score)
            result_summary[window]['bps'].append(bps.sum())
            
        result_summary[window]['all_removed_tiles'] = all_removed_tiles
        insert_coords = test_coords.copy()
        result_summary[window]['insert_coords'] = insert_coords
    return result_summary


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



