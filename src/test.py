from pixel import PIXELConfig, PIXELForPreTraining

config = PIXELConfig.from_pretrained("Team-PIXEL/pixel-base")
model = PIXELForPreTraining.from_pretrained("Team-PIXEL/pixel-base", config=config)
