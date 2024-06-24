import subprocess
import pandas as pd
import os, sys
from creme import utils


def main():

    cell_lines = sys.argv[1]

    for cell_line in cell_lines.split(','):
        for epigenetic_assay in ['tf', 'accessibility', 'histone']:

            biochem_dir = utils.make_dir(f"../data/biochemical_marks")
            cell_line_dir = utils.make_dir(f'{biochem_dir}/{cell_line}')
            outdir = utils.make_dir(f"{cell_line_dir}/{epigenetic_assay}")
            metadata = pd.read_csv('../results/metadata.tsv', sep='\t')
            nondnase_epi_tracks = metadata.query('`Output type` == "fold change over control" & `File analysis status` == "released" & \
                            `File assembly` == "GRCh38" ')
            dnase_epi_tracks = metadata.query('`File type` == "bigWig" & `File analysis status` == "released" & \
                            `File assembly` == "GRCh38" & Assay == "DNase-seq"')

            epi_tracks = pd.concat([nondnase_epi_tracks, dnase_epi_tracks])
            epi_tracks = epi_tracks[epi_tracks['Biosample term name'] == cell_line]
            print(epi_tracks.shape)
            if epigenetic_assay == 'histone':
                epi_tracks = epi_tracks[epi_tracks['Assay'] == "Histone ChIP-seq"]
            elif epigenetic_assay == 'accessibility':
                print(epi_tracks['Assay'].unique())
                epi_tracks = epi_tracks[(epi_tracks['Assay'] == 'ATAC-seq') | (epi_tracks['Assay'] == 'DNase-seq')]
            elif epigenetic_assay.lower() == 'tf':
                epi_tracks = epi_tracks[epi_tracks['Assay'] == "TF ChIP-seq"]


            epi_tracks['replicate_name_length'] = [len(t) for t in epi_tracks['Technical replicate(s)']]
            choose_best_repl = []
            for _, df in epi_tracks.groupby('Experiment accession'):
                df = pd.DataFrame(df.sort_values('replicate_name_length').iloc[-1, :]).T
                choose_best_repl.append(df)
            epi_tracks = pd.concat(choose_best_repl)

            # open file
            text_path = f'../data/{cell_line}.txt'
            n_files = 0
            with open(text_path, 'w+') as f:
                # write elements of list
                for items in epi_tracks['File download URL']:
                    if not os.path.isfile(f'{outdir}/{items.split("/")[-1]}'):
                        f.write('%s\n' % items)
                        n_files += 1

            epi_tracks.to_csv(f'{outdir}/metadata.csv')
            print(f'Downloading {n_files} files')
            if n_files > 0:
                cmd = f'cat {text_path} | xargs -n 16 -P 2 wget -q -P {outdir}'
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()



if __name__ == '__main__':
    main()