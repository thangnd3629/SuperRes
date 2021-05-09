import torch
import torch.nn as nn 



class RDB(nn.Module):
    def __init__(self,in_channels,latent_channel, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * latent_channel, latent_channel, 3))
        #local fusion
        self.lff = nn.Conv2d(in_channels + latent_channel * num_layers, in_channels, kernel_size=1)
    def _make_layer(self, in_channels, out_channels, kernel_size):
        return nn.ModuleDict({
            'conv':   nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2) # same size conv 
            ,
            'relu':nn.ReLU(inplace=True)
        })
         
            
            
        
    def forward(self,x):
        input_map = x
        for i in range(self.num_layers):
            y = self.layers[i]['conv'](input_map)
            y = self.layers[i]['relu'](y)
            input_map = torch.cat([input_map, y], dim=1)
        return x + self.lff(input_map)


class ResidualDenseNet(nn.Module):
    def __init__(self, scale, shallow_feature, growth_rate, num_blocks, num_layers ):
        super().__init__()
        self.scale = scale
        self.num_sf = shallow_feature
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.num_layers = num_layers # number of layer inside RDB

        self.sfe1 = nn.Conv2d(3, shallow_feature, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv2d(shallow_feature, shallow_feature, kernel_size=3, padding=1)
        #first block
        self.rdbs = nn.ModuleList([RDB(shallow_feature,growth_rate,num_layers )])
        for i in range(num_blocks-1):
            self.rdbs.append(RDB(growth_rate, growth_rate,num_layers))
        self.gff = nn.Sequential(*[
            nn.Conv2d(growth_rate * num_blocks, shallow_feature, kernel_size=1),
            nn.Conv2d(shallow_feature, shallow_feature, kernel_size=3, padding=1)
        ])

        #upsample
        assert scale == 2 , 'not supported scale '
        self.upscale = nn.Sequential(
            nn.Conv2d(shallow_feature, shallow_feature * 2 * 2 , kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=scale)
        )
        self.output = nn.Conv2d(shallow_feature, 3, kernel_size=3,padding=1)

    def forward(self,x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        local = sfe2
        local_dense = []
        for i in range(self.num_blocks):
            local = self.rdbs[i](local)
            local_dense.append(local)
        gff = self.gff(torch.cat(local_dense, dim=1))
        df = sfe1 + gff
        upsample = self.upscale(df)
        output = self.output(upsample)
        return output


