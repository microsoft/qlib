export interface PrepareBoxplotDataOpt {
    boundIQR?: number | 'none';
    itemNameFormatter?: string | ((params: {
        value: number;
    }) => string);
}
/**
 * See:
 *  <https://en.wikipedia.org/wiki/Box_plot#cite_note-frigge_hoaglin_iglewicz-2>
 *  <http://stat.ethz.ch/R-manual/R-devel/library/grDevices/html/boxplot.stats.html>
 *
 * Helper method for preparing data.
 *
 * @param rawData like
 *        [
 *            [12,232,443], (raw data set for the first box)
 *            [3843,5545,1232], (raw data set for the second box)
 *            ...
 *        ]
 * @param opt.boundIQR=1.5 Data less than min bound is outlier.
 *      default 1.5, means Q1 - 1.5 * (Q3 - Q1).
 *      If 'none'/0 passed, min bound will not be used.
 */
export default function prepareBoxplotData(rawData: number[][], opt: PrepareBoxplotDataOpt): {
    boxData: (number | string)[][];
    outliers: (number | string)[][];
};
