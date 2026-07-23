import SeriesData from '../data/SeriesData.js';
/**
 * LegendVisualProvider is an bridge that pick encoded color from data and
 * provide to the legend component.
 */
declare class LegendVisualProvider {
    private _getDataWithEncodedVisual;
    private _getRawData;
    constructor(getDataWithEncodedVisual: () => SeriesData, getRawData: () => SeriesData);
    getAllNames(): string[];
    containName(name: string): boolean;
    indexOfName(name: string): number;
    getItemVisual(dataIndex: number, key: string): any;
}
export default LegendVisualProvider;
