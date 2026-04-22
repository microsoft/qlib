import { DefaultDataVisual } from '../data/SeriesData.js';
export interface LineDataVisual extends DefaultDataVisual {
    fromSymbol: string;
    toSymbol: string;
    fromSymbolSize: number | number[];
    toSymbolSize: number | number[];
    fromSymbolRotate: number;
    toSymbolRotate: number;
    fromSymbolOffset: string | number | (string | number)[];
    toSymbolOffset: string | number | (string | number)[];
    fromSymbolKeepAspect: boolean;
    toSymbolKeepAspect: boolean;
}
