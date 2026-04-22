import type Model from '../../model/Model.js';
import type Sector from 'zrender/lib/graphic/shape/Sector.js';
export declare function getSectorCornerRadius(model: Model<{
    borderRadius?: string | number | (string | number)[];
}>, shape: Pick<Sector['shape'], 'r0' | 'r'>, zeroIfNull?: boolean): {
    cornerRadius: number;
} | {
    cornerRadius: number[];
};
