import Geo from './Geo.js';
export default function geoPrepareCustom(coordSys: Geo): {
    coordSys: {
        type: string;
        x: number;
        y: number;
        width: number;
        height: number;
        zoom: number;
    };
    api: {
        coord: (data: number[]) => number[];
        size: (dataSize: number[], dataItem: number[]) => number[];
    };
};
