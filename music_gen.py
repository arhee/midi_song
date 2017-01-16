tsteps = 100

start_ix = random.randint(0, steps-maxlen - 1)
psg_seed = music_mat[start_ix:start_ix+maxlen,:]
x = psg_seed[None,:,:]


new_song = []
new_song.append(x)

for _ in range(tsteps):

    preds = model.predict(x, verbose=0)[0]

    # this is a timestep slice
    new_note = np.zeros(nnotes)
    new_note[preds.argmax()] = 1
    new_note = new_note[None, None,:]

    # redo the seed psg
    psg = np.concatenate([x,new_note], axis=1)
    x = psg[:, 1:,:]

    new_song.append(new_note)

new_song = np.concatenate(new_song, axis=1)
new_song = np.squeeze(new_song)