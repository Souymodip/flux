import flux.util as util

def test():
    name = "flux-schnell"
    device = "cpu"
    ae_model = util.load_ae(name, device)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    test()
